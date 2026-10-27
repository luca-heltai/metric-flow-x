#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate a completed ten-cycle ADAVN run and summarize cycles 7--10.

Uses the consolidated midpoint CSV in SI units. Requires the tutorial's
0.02 s output cadence (40 intervals per 0.8 s cycle).
"""
import argparse
import csv
import json
import math
from pathlib import Path

from analyze_full import MMHG_PA


def summarize(run):
    initialization = json.loads((run/'initialization_statistics.json').read_text())
    if not initialization['tolerance_ramp_enabled'] or any(
            initialization[key] != 1 for key in
            ('ramp_last_relative_factor', 'ramp_last_absolute_factor')):
        raise ValueError('The run must reach final tolerances after startup')
    with (run/'ida_statistics.csv').open() as stream:
        receipts = list(csv.DictReader(stream))
    if len(receipts) != 1 or int(receipts[0]['n_steps']) < 1:
        raise ValueError('Use a fresh directory with one completed integration receipt')
    with (run/'probes.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    fields = ('pressure_Pa', 'flow_m3_s', 'area_m2', 'velocity_m_s')
    probes = {}
    for row in rows:
        vid, time = int(row['vessel_id']), float(row['time_s'])
        values = [float(row[field]) for field in fields]
        if not math.isfinite(time) or not all(map(math.isfinite, values)) or values[2] <= 0:
            raise ValueError('Nonfinite or nonpositive probe state')
        series = probes.setdefault(vid, {})
        tick = round(time/.02)
        if abs(time-tick*.02) > 1e-9 or tick in series:
            raise ValueError('Unexpected output cadence or duplicate probe sample')
        series[tick] = values
    if not {1630, 1631}.issubset(probes):
        raise ValueError('Both halves of aortic source row 815 must be probed')
    for series in probes.values():
        if set(series) != set(range(401)):
            raise ValueError('Expected the complete ten cycles, t=0 through t=8 s')
    aorta = {tick: [(probes[1630][tick][j]+probes[1631][tick][j])/2
                    for j in range(len(fields))] for tick in range(401)}
    cycles = []
    for cycle in range(7, 11):
        lo, hi = (cycle-1)*40, cycle*40
        means = [sum((aorta[k][j]+aorta[k+1][j])/2 for k in range(lo, hi))/40
                 for j in range(len(fields))]
        cycles.append(dict(cycle=cycle, MAP_mmHg=means[0]/MMHG_PA,
                           mean_aortic_flow_ml_s=means[1]*1e6,
                           systolic_pressure_mmHg=max(aorta[k][0] for k in range(lo, hi+1))/MMHG_PA,
                           diastolic_pressure_mmHg=min(aorta[k][0] for k in range(lo, hi+1))/MMHG_PA))
    central_ids = (8298, 8299, 8574, 8575, 8696, 8697)
    if set(central_ids).issubset(probes):
        for cycle in cycles:
            lo, hi = (cycle['cycle']-1)*40, cycle['cycle']*40
            cycle['mean_total_central_probe_flow_ml_s'] = sum(
                sum((probes[vid][k][1]+probes[vid][k+1][1])/2
                    for k in range(lo, hi))/40 for vid in central_ids)/2*1e6
    periodicity = {}
    for vid, series in probes.items():
        periodicity[str(vid)] = {field: max(abs(series[k][j]-series[k-40][j])
                                           for k in range(360, 401))
                                 for j, field in enumerate(fields)}
    report = dict(period_s=.8, final_time_s=8., startup_cycles=6,
                  final_tolerance_cycles=cycles, cycle10_MAP_relative_difference=(cycles[-1]['MAP_mmHg']-92.58)/92.58,
                  paper_Table3_MAP_mmHg=92.58, cycle10_vs9_max_absolute_difference=periodicity,
                  pressure_sampling='Spatial mean of midpoint probes in the two halves of source row 815.',
                  interpretation='Prescribed-flow open-loop elastic reduction; CO and reservoir CVP are imposed. Ten cycles do not by themselves establish periodic convergence.',
                  native_statistics=receipts[0], initialization=initialization)
    mass_file = run/'mass_conservation.csv'
    if mass_file.exists():
        with mass_file.open() as stream:
            mass = {round(float(row['t'])/.02): row for row in csv.DictReader(stream)}
        if not set(range(360, 401)).issubset(mass):
            raise ValueError('Incomplete final-cycle volume diagnostics')
        volume_rate = (float(mass[400]['total_volume'])-float(mass[360]['total_volume']))/.8
        flux = sum((float(mass[k]['net_flux'])+float(mass[k+1]['net_flux']))/2
                   for k in range(360, 400))/40
        if not all(map(math.isfinite, (volume_rate, flux))):
            raise ValueError('Nonfinite volume diagnostics')
        report['cycle10_mesh_volume_balance'] = dict(
            volume_change_rate_ml_s=volume_rate*1e6,
            trapezoidal_mean_net_boundary_flux_ml_s=flux*1e6,
            mismatch_ml_s=(volume_rate-flux)*1e6,
            mismatch_fraction_of_imposed_mean_flow=(volume_rate-flux)/96.36e-6,
            interpretation='Mesh vessel volume only; all mesh boundary ports, including exchanges with lumped compartments. Output cadence 0.02 s; finite-cadence check, not an exact residual identity.')
    (run/'cycle_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    summarize(parser.parse_args().run_dir)
