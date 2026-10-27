#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Recover all ADAVN vessel coordinates from the arterial and venous atlases.

Requires NumPy/SciPy for globally consistent binary surface orientations.
The surface-derived geometry is subject to the separate HeMoLab terms.
"""
import hashlib
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from recover_geometry import sample_curve, surface_centers


def ordered_groups(rows):
    """Retain source-row identities while sorting valve/SR split chains."""
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[re.sub(r'_\d+$', '', row[0])].append((index, row))
    ordered = {}
    for name, entries in groups.items():
        outgoing = {}
        ends = set()
        for index, row in entries:
            start, end = int(row[1]), int(row[2])
            if start in outgoing:
                raise ValueError(name+': branching within a named surface')
            outgoing[start] = (index, row)
            ends.add(end)
        starts = set(outgoing)-ends
        if len(starts) != 1:
            raise ValueError(name+': expected one directed source chain')
        node = starts.pop()
        chain = []
        while node in outgoing:
            entry = outgoing.pop(node)
            chain.append(entry)
            node = int(entry[1][2])
        if outgoing:
            raise ValueError(name+': disconnected named source chain')
        ordered[name] = chain
    return ordered


def orient(candidates, incident):
    """Minimize squared shared-node disagreement over both surface directions.

    One binary variable per named vessel. A continuous product variable for
    each interacting pair is constrained by its exact binary linearization.
    No assumed flow direction or greedy seed fixes an anatomical orientation.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    names = sorted(candidates)
    number = {name: i for i, name in enumerate(names)}
    costs = {}
    for node, vessels in incident.items():
        for left, right in itertools.combinations(sorted(vessels), 2):
            values = costs.setdefault((left, right), [0.]*4)
            for a, b in itertools.product(range(2), repeat=2):
                values[a*2+b] += math.dist(candidates[left][a][node],
                                          candidates[right][b][node])**2
    pairs = sorted(costs)
    n = len(names)
    objective = np.zeros(n+len(pairs))
    row, col, data, upper = [], [], [], []
    for edge, (left, right) in enumerate(pairs):
        i, j, z = number[left], number[right], n+edge
        c00, c01, c10, c11 = costs[left, right]
        objective[i] += c10-c00
        objective[j] += c01-c00
        objective[z] = c11-c10-c01+c00
        for entries, limit in [([(z, 1.), (i, -1.)], 0.),
                                ([(z, 1.), (j, -1.)], 0.),
                                ([(i, 1.), (j, 1.), (z, -1.)], 1.)]:
            k = len(upper)
            for index, value in entries:
                row.append(k); col.append(index); data.append(value)
            upper.append(limit)
    constraints = ()
    if pairs:
        matrix = coo_matrix((data, (row, col)), shape=(len(upper), len(objective))).tocsc()
        constraints = LinearConstraint(matrix, -np.inf, upper)
    result = milp(objective, integrality=np.r_[np.ones(n), np.zeros(len(pairs))],
                  bounds=Bounds(0., 1.), constraints=constraints,
                  options={'time_limit': 120., 'mip_rel_gap': 1e-10})
    if not result.success:
        raise ValueError('Anatomical orientation optimization failed: '+result.message)
    flips = {name: int(result.x[i] > .5) for name, i in number.items()}
    return flips, dict(method='global binary shared-node least squares (HiGHS MILP)',
                       named_vessels=n, interacting_pairs=len(pairs),
                       optimal=True, mip_gap=float(getattr(result, 'mip_gap', 0.)))


def reconcile(rows, points, max_iterations=20000):
    """Project anatomical shared nodes onto all source chord-length bounds.

    Dykstra corrections retain the least-displacement objective instead of
    repeatedly shortening a chain without accounting for earlier moves.
    Coordinates and source lengths are in metres.
    """
    original = dict(points)
    edges = [(int(row[1]), int(row[2]), float(row[3])*.01*(1-1e-9)) for row in rows]
    correction = [(0., 0., 0.) for _ in edges]
    for iteration in range(max_iterations):
        for k, (a, b, length) in enumerate(edges):
            pa = tuple(points[a][j]+correction[k][j] for j in range(3))
            pb = tuple(points[b][j]-correction[k][j] for j in range(3))
            delta = tuple(pb[j]-pa[j] for j in range(3))
            distance = math.sqrt(sum(x*x for x in delta))
            factor = max(0., (distance-length)/(2*distance)) if distance else 0.
            shift = tuple(factor*x for x in delta)
            points[a] = tuple(pa[j]+shift[j] for j in range(3))
            points[b] = tuple(pb[j]-shift[j] for j in range(3))
            correction[k] = tuple(-x for x in shift)
        if iteration % 10 == 0:
            violation = max((math.dist(points[a], points[b])-length)/length
                            for a, b, length in edges)
            if violation < 1e-10:
                break
    else:
        raise ValueError('Source-length reconciliation did not converge')
    changes = [math.dist(points[node], original[node]) for node in points]
    return dict(iterations=iteration+1, max_relative_chord_violation=max(0., violation),
                max_node_adjustment_m=max(changes),
                rms_node_adjustment_m=math.sqrt(sum(x*x for x in changes)/len(changes)))


def recover(rows, root, report_path):
    groups = ordered_groups(rows)
    manifest = json.loads(Path(__file__).with_name('sources.json').read_text())
    candidates, curves, fractions, incident = {}, {}, {}, defaultdict(set)
    source_hashes, lengths = {}, []
    for name, chain in groups.items():
        artery = chain[0][1][6] == '1'
        atlas = manifest['atlas' if artery else 'venous_atlas']
        directory = 'surfaces' if artery else 'venous_surfaces'
        path = root/'source'/directory/('SW_'+name+'.vtk')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != atlas['files'][name]:
            raise ValueError('Anatomical surface checksum mismatch: '+str(path))
        source_hashes[directory+'/'+path.name] = digest
        curve, _ = surface_centers(path)
        curves[name] = curve
        total = sum(float(row[3]) for _, row in chain)
        node_ids = [int(chain[0][1][1])]+[int(row[2]) for _, row in chain]
        part = [0.]
        for _, row in chain:
            part.append(part[-1]+float(row[3])/total)
        fractions[name] = part
        candidates[name] = [dict(zip(node_ids, (sample_curve(curve, min(1., f)) for f in part))),
                            dict(zip(node_ids, (sample_curve(curve, max(0., 1-f)) for f in part)))]
        for node in node_ids:
            incident[node].add(name)
        lengths.append(dict(name=name, source_length_cm=total,
                            surface_centerline_length_cm=sum(math.dist(a, b) for a, b in zip(curve, curve[1:])),
                            source_rows=[index for index, _ in chain]))
    flips, optimization = orient(candidates, incident)
    points, disagreements, hints = {}, [], {}
    for node, names in incident.items():
        samples = [candidates[name][flips[name]][node] for name in sorted(names)]
        points[node] = tuple(sum(p[j] for p in samples)/len(samples)*.01 for j in range(3))
        spread = max(math.dist(a, b) for a in samples for b in samples)*.01
        disagreements.append(dict(node=node, spread_m=spread, vessels=sorted(names)))
    adjustment = reconcile(rows, points)
    for name, chain in groups.items():
        part = fractions[name]
        for j, (index, _) in enumerate(chain):
            f = (part[j]+part[j+1])/2
            if flips[name]:
                f = 1-f
            hints[index] = tuple(x*.01 for x in sample_curve(curves[name], f))
    report = dict(mode='anatomical', units='m', named_vessels=len(groups),
                  named_arteries=sum(chain[0][1][6]=='1' for chain in groups.values()),
                  named_veins=sum(chain[0][1][6]=='0' for chain in groups.values()),
                  optimization=optimization, source_lengths=lengths,
                  orientation_reversed=sorted(name for name in flips if flips[name]),
                  node_disagreements=sorted(disagreements, key=lambda x:x['spread_m'], reverse=True),
                  length_reconciliation=adjustment, source_sha256=source_hashes,
                  points_m=points,
                  description='Surface-derived anatomical reconstruction, not the original unpublished splines; shared-node averaging and source-length reconciliation are explicit adaptations.',
                  license=manifest['atlas']['license'], terms_url=manifest['atlas']['terms_url'])
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    return points, hints, report
