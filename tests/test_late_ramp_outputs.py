"""Offline physical-comparator and preparation checks; no solver is launched."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1]
TOOLS = SOURCE / "tools"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LateRampOutputs(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="metric-flow-late-validator-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.reference, self.actual = self.root / "reference", self.root / "candidate"
        self.bench = load_module("benchmark_ida_ramp")
        self.validator = load_module("compare_late_ramp_outputs")
        self.policy = json.loads((TOOLS / "ida_ramp_policy.json").read_text())
        self.physics = self.validator.load_physics(TOOLS / "compare_inflow_trajectories.py", self.policy)
        self.material = self.physics.read_vtk(SOURCE / "parameters/single_vessel.vtk")
        self.area, self.velocity = self.material["a_d"], .01
        self.flow = self.area * self.velocity
        self.pressure = self.material["p0"] + self.material["p_d"]
        self.times = [i * .05 for i in range(201)]
        self.fixture(self.reference, "reference")
        self.fixture(self.actual, "uniform")

    @staticmethod
    def write_csv(path, fields, rows):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def fixture(self, folder, variant):
        folder.mkdir(exist_ok=True)
        output = folder / "output"
        output.mkdir(exist_ok=True)
        (folder / "last_used_parameters.prm").write_text(self.bench.parameters(SOURCE, output, self.policy, variant))
        rows = [dict(time_s=t, P_dynpcm2=self.pressure * 10, Q_cm3ps=self.flow * 1e6,
                     A_cm2=self.area * 1e4, U_cmps=self.velocity * 100, boundary_1_label="boundary_1") for t in self.times]
        self.write_csv(output / "HDG_IDA_Vessel_0.csv", list(rows[0]), rows)
        for boundary in (0, 1):
            rows = [dict(time_s=t, pressure_Pa=self.pressure, flow_m3_s=self.flow if boundary else -self.flow,
                         boundary_id=boundary, vessel_id=0, label=f"boundary_{boundary}") for t in self.times]
            self.write_csv(output / f"terminal_{boundary}.csv", list(rows[0]), rows)
        rows = [dict(t=t, total_volume=self.material["L"] * self.area, dVdt=0, net_flux=0, imbalance=0, rel_imbalance=0) for t in self.times]
        self.write_csv(output / "mass_conservation.csv", list(rows[0]), rows)
        rows = [dict(boundary_id=1, vessel_id=0, label="boundary_1", R1=1e7, R2=1e8, C=1e-8, P_out=1000)]
        self.write_csv(output / "network_terminals.csv", list(rows[0]), rows)

    def compare(self):
        return self.validator.compare(self.reference, self.actual, self.policy, self.physics)

    def mutate_velocity(self, index, factor):
        path = self.actual / "output/HDG_IDA_Vessel_0.csv"
        fields, rows = self.physics.read_csv(path)
        value = self.velocity + factor * (self.policy["base_atol"] + self.policy["base_rtol"] * abs(self.velocity))
        rows[index]["U_cmps"], rows[index]["Q_cm3ps"] = str(value * 100), str(self.area * value * 1e6)
        self.write_csv(path, fields, rows)

    def test_campaign_sizes_and_balanced_order(self):
        self.assertEqual(self.bench.campaign_plan(self.policy, True), (["uniform-0", "ramp-0", "reference"], 1, "correctness_only"))
        names, pairs, kind = self.bench.campaign_plan(self.policy, False)
        self.assertEqual((len(names), pairs, kind), (9, 4, "formal_performance"))
        self.assertEqual(sum(names[i].startswith("uniform") for i in range(0, 8, 2)), 2)

    def validation_results(self, n_pairs=1):
        comparison = self.compare()
        return {f'{kind}-{i}':json.loads(json.dumps(comparison))
                for i in range(n_pairs) for kind in ('uniform', 'ramp', 'pair')}

    @staticmethod
    def fail_metric(comparison):
        comparison['metrics']['velocity'].update(max_normalized_error=25.0, accepted=False)
        comparison['accepted'] = False

    def test_v3_keeps_failed_reference_gate(self):
        validations = self.validation_results()
        self.fail_metric(validations['uniform-0'])
        old_policy = {key:value for key,value in self.policy.items()
                      if key not in ('acceptance_target', 'mandatory_comparisons', 'diagnostic_comparisons', 'absolute_accuracy_certified')}
        result = self.bench.acceptance_summary(validations, old_policy, 1)
        self.assertFalse(result['accepted'])
        self.assertTrue(result['equivalence_passed'])
        self.assertFalse(result['tighter_reference_diagnostics_passed'])

    def test_v4_pair_accepts_without_absolute_accuracy_claim(self):
        validations = self.validation_results(4)
        self.fail_metric(validations['uniform-0'])
        self.fail_metric(validations['ramp-3'])
        result = self.bench.acceptance_summary(validations, self.policy, 4)
        self.assertTrue(result['accepted'])
        self.assertTrue(result['equivalence_passed'])
        self.assertFalse(result['tighter_reference_diagnostics_passed'])
        self.assertFalse(result['absolute_accuracy_certified'])
        self.assertFalse(validations['uniform-0']['accepted'])

    def test_v4_every_pair_must_pass(self):
        validations = self.validation_results(4)
        self.fail_metric(validations['pair-3'])
        result = self.bench.acceptance_summary(validations, self.policy, 4)
        self.assertFalse(result['accepted'])
        self.assertFalse(result['equivalence_passed'])
        self.assertTrue(result['tighter_reference_diagnostics_passed'])

    def test_missing_comparisons_rejected(self):
        for name in ('uniform-0', 'ramp-0', 'pair-0'):
            with self.subTest(missing=name):
                validations = self.validation_results()
                del validations[name]
                with self.assertRaises(ValueError):
                    self.bench.acceptance_summary(validations, self.policy, 1)

    def test_unknown_acceptance_mode_rejected(self):
        with self.assertRaises(ValueError):
            self.bench.acceptance_summary(self.validation_results(), dict(self.policy, acceptance_target='unknown'), 1)

    def test_diagnostic_nonfinite_and_inconsistent_metrics_rejected(self):
        for error in (float('nan'), 1.0):
            with self.subTest(error=error):
                validations = self.validation_results()
                validations['uniform-0']['metrics']['velocity']['max_normalized_error'] = error
                with self.assertRaises(ValueError):
                    self.bench.acceptance_summary(validations, self.policy, 1)

    def test_exact(self):
        self.assertTrue(self.compare()["accepted"])

    def test_subweight_late_perturbation(self):
        self.mutate_velocity(180, .5)
        result = self.compare()
        self.assertTrue(result["accepted"])
        self.assertAlmostEqual(result["metrics"]["velocity"]["max_normalized_error"], .5, places=5)

    def test_superweight_late_perturbation(self):
        self.mutate_velocity(180, 2)
        self.assertFalse(self.compare()["accepted"])

    def test_early_transient_excluded(self):
        self.fixture(self.actual, "ramp")
        self.mutate_velocity(10, 1000)
        self.assertTrue(self.compare()["accepted"])

    def test_changed_parameters_rejected(self):
        for old, new in (("set Viscosity coefficient (mu) = 0.004", "set Viscosity coefficient (mu) = 0.005"),
                         ("set Absolute error tolerance = 1e-12", "set Absolute error tolerance = 1e-10"),
                         ("set Tolerance ramp initial relative factor = 1000.0", "set Tolerance ramp initial relative factor = 2000.0"),
                         ("set Maximum IDA internal steps per output = 10000", "set Maximum IDA internal steps per output = 20000")):
            with self.subTest(parameter=old):
                self.fixture(self.actual, "ramp")
                path = self.actual / "last_used_parameters.prm"
                self.assertIn(old, path.read_text())
                path.write_text(path.read_text().replace(old, new))
                with self.assertRaises(ValueError):
                    self.compare()

    def test_terminal_derived_state_difference(self):
        path = self.actual / "output/terminal_1.csv"
        fields, rows = self.physics.read_csv(path)
        rows[180]["pressure_Pa"] = str(self.pressure + 1)
        self.write_csv(path, fields, rows)
        self.assertFalse(self.compare()["accepted"])

    def test_nonfinite_output(self):
        path = self.actual / "output/mass_conservation.csv"
        fields, rows = self.physics.read_csv(path)
        rows[180]["total_volume"] = "nan"
        self.write_csv(path, fields, rows)
        with self.assertRaises(ValueError):
            self.compare()

    def test_pair_timestamp_bound_unchanged(self):
        path = self.actual / "output/terminal_1.csv"
        fields, rows = self.physics.read_csv(path)
        rows[180]["time_s"] = str(float(rows[180]["time_s"]) + 5e-14)
        self.write_csv(path, fields, rows)
        with self.assertRaises(ValueError):
            self.compare()

    def test_negative_ida_status_in_log_is_rejected(self):
        log = self.root / 'run.log'
        for token in ('[IDAS ERROR]', '[IDA ERROR]'):
            with self.subTest(token=token):
                log.write_text(token+' IDASolve: maximum internal steps exhausted')
                with self.assertRaises(RuntimeError):
                    self.bench.check_solver_log(log)
        log.write_text('Normal IDA completion')
        self.bench.check_solver_log(log)

    def test_cli_prepares_only_from_another_directory(self):
        output = self.root / "prepared"
        subprocess.run([sys.executable, str(TOOLS / "benchmark_ida_ramp.py"), "--executable", sys.executable,
                        "--output", str(output), "--pilot"], cwd=self.root, capture_output=True, text=True, check=True)
        protocol = json.loads((output / "protocol.json").read_text())
        self.assertFalse(protocol["executed"])
        self.assertEqual(protocol["measurement_kind"], "correctness_only")
        self.assertEqual(len(protocol["cases"]), 3)
        self.assertFalse(any(output.rglob("run.log")))

    def test_comparator_cli_uses_repository_defaults(self):
        report = self.root / "comparison.json"
        subprocess.run([sys.executable, str(TOOLS / "compare_late_ramp_outputs.py"), "--before", str(self.reference),
                        "--after", str(self.actual), "--output", str(report)], cwd=self.root, capture_output=True, text=True, check=True)
        self.assertTrue(json.loads(report.read_text())["accepted"])


if __name__ == "__main__":
    unittest.main()
