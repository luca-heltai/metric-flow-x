# SPDX-License-Identifier: MIT
"""Offline checks of the coupled graph, taper units and full metric geometry."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/adan"))
import prepare_full
from analyze_full import partial_resistance_fraction
from build_vtk import midpoint


def row(name, a, b, kind, radius=.1, resistance="-", connections=("-", "-"), perforator=-1):
    return list(map(str, [name, a, b, 2, radius, radius, kind, 1, perforator,
                         resistance, 1e-8, 20, *connections, ""]))


class FullAdavnTests(unittest.TestCase):
    def fixture(self):
        # Branch 1->2 has two venous drains of radii 1:2. Branch 3->1 has
        # its terminal at its first node. Repeated venous names are preserved.
        return [row("arterial_root", 0, 1, 1),
                row("artery1", 1, 2, 1, resistance=9, connections=("10", "11")),
                row("artery2", 3, 1, 1, resistance=10, connections=("10", "-")),
                row("vein", 4, 5, 0, radius=.1, resistance=1, connections=("10", "-")),
                row("vein", 6, 5, 0, radius=.2, resistance=1, connections=("11", "-")),
                row("venous_root", 5, 7, 0)]

    def test_connections_and_parallel_resistance(self):
        rows = self.fixture()
        edges, inlet, drains, _ = prepare_full.hydraulic_graph(rows)
        self.assertEqual((inlet, drains), (0, [7]))
        self.assertEqual([(e["node0"], e["node1"]) for e in edges[6:]],
                         [(8, 4), (9, 6), (2, 10), (10, 8), (10, 9), (3, 11), (11, 8)])
        # Independent series/parallel expectation, including CGS->SI.
        self.assertAlmostEqual(edges[8]["resistance_Pa_s_m3"], .15*9e5)
        self.assertAlmostEqual(edges[9]["resistance_Pa_s_m3"], .85*81e5)
        self.assertAlmostEqual(edges[10]["resistance_Pa_s_m3"], .85*81e5/8)
        equivalent = edges[8]["resistance_Pa_s_m3"]+1/sum(
            1/e["resistance_Pa_s_m3"] for e in edges[9:11])
        self.assertAlmostEqual(equivalent, 9e5)

    def test_venous_compartments(self):
        data = prepare_full.venous_compartments(self.fixture(), 2.)
        self.assertAlmostEqual(data[3]["resistance"], 1040*2/(math.pi*.001**2))
        self.assertAlmostEqual(data[4]["resistance"], data[3]["resistance"]/4)
        self.assertAlmostEqual(data[3]["compliance"]/data[4]["compliance"], 5/4)
        volume = math.pi*.02*(.001**2+.002**2+.001**2)
        target = 146e-6/133.322435415-volume/(1040*2**2)
        self.assertAlmostEqual(sum(row["compliance"] for row in data.values())/target, 1.)

    def test_viscosity_and_taper_integral(self):
        tube = row("tube", 0, 1, 1)
        expected = 8*.004*.02/(math.pi*.001**4)
        self.assertAlmostEqual(prepare_full.tube_resistance(tube)/expected, 1.)
        tube[8] = "1"
        self.assertAlmostEqual(prepare_full.tube_resistance(tube)/expected, .25)
        tube[6] = "0"
        self.assertAlmostEqual(prepare_full.tube_resistance(tube)/expected, 1.)
        self.assertAlmostEqual(partial_resistance_fraction(1., 2., .5), 152/189)
        self.assertAlmostEqual(partial_resistance_fraction(1., 1., .5), .5)

    def test_geometry_keeps_every_source_length(self):
        rows = self.fixture()
        _, inlet, drains, _ = prepare_full.hydraulic_graph(rows)
        nodes = prepare_full.metric_geometry(rows, inlet, drains)
        self.assertEqual(set(nodes), set(range(8)))
        for source in rows:
            a, b = (nodes[int(source[j])] for j in (1, 2))
            length = float(source[3])*.01
            mid = midpoint(a, b, length, (0., 0., 0.))
            self.assertAlmostEqual(math.dist(a, mid), length/2)
            self.assertAlmostEqual(math.dist(mid, b), length/2)

    def test_unknown_connection_rejected(self):
        rows = self.fixture()
        rows[1][12] = "missing"
        with self.assertRaises(KeyError):
            prepare_full.hydraulic_graph(rows)


if __name__ == "__main__":
    unittest.main()
