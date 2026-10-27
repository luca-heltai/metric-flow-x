#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Anatomical chain ordering, global orientation and source-length constraints."""
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/adan'))
from recover_full import ordered_groups, orient, reconcile
import download_sources


def segment(name, a, b, length):
    return [name, str(a), str(b), str(length), '.1', '.1', '0']


class AnatomicalNetwork(unittest.TestCase):
    def test_split_chains_are_ordered_by_connectivity(self):
        rows = [segment('V_vein_L_4041', 4, 9, 1),
                segment('V_vein_L_4042', 10, 6, 2),
                segment('V_vein_L_4041', 9, 10, .5)]
        groups = ordered_groups(rows)
        self.assertEqual([i for i, _ in groups['V_vein_L']], [0, 2, 1])

    def test_branch_within_named_surface_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'branching'):
            ordered_groups([segment('V_vein_0', 1, 2, 1),
                            segment('V_vein_1', 1, 3, 1)])

    def test_global_orientation_recovers_shared_joints(self):
        # Three chains form a loop. Each correct direction independently gives
        # the same anatomical point at its two shared source nodes. Raw B/C
        # surfaces are reversed relative to the source connectivity.
        a, b, c = (0., 0., 0.), (3., 0., 0.), (1., 2., 0.)
        candidates = {'A': [{0:a, 1:b}, {0:b, 1:a}],
                      'B': [{1:c, 2:b}, {1:b, 2:c}],
                      'C': [{2:a, 0:c}, {2:c, 0:a}]}
        flips, result = orient(candidates, {0:{'A','C'}, 1:{'A','B'}, 2:{'B','C'}})
        self.assertEqual(flips, {'A':0, 'B':1, 'C':1})
        self.assertTrue(result['optimal'])

    def test_length_projection_retains_anatomical_position(self):
        rows = [segment('V_a_0', 0, 1, 1)]  # one centimetre
        points = {0:(2., 3., 4.), 1:(2.012, 3., 4.)}
        result = reconcile(rows, points)
        # Independent Euclidean projection moves each end by 1 mm, keeps
        # the absolute anatomical midpoint, and limits the chord to 1 cm.
        self.assertAlmostEqual(points[0][0], 2.001, places=10)
        self.assertAlmostEqual(points[1][0], 2.011, places=10)
        self.assertAlmostEqual((points[0][0]+points[1][0])/2, 2.006)
        self.assertLessEqual(math.dist(points[0], points[1]), .01)
        self.assertAlmostEqual(result['max_node_adjustment_m'], .001, places=10)

    def test_venous_download_keeps_explicit_permission_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'written permission'):
                download_sources.fetch(Path(folder), include_veins=True)


if __name__ == '__main__':
    unittest.main()
