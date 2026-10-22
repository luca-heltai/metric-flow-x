#!/usr/bin/env python3
import hashlib
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import collect_convergence  # noqa: E402
import generate_mms_expressions  # noqa: E402


class ConvergenceTests(unittest.TestCase):
    def test_reference_is_parsed_without_rounding(self):
        path = ROOT / "tutorials/01_single_vessel_mms/reference/convergence.txt"
        text = path.read_text(encoding="utf-8")
        document = collect_convergence.canonical_document(text, str(path))
        self.assertEqual([t["polynomial_degree"] for t in document["tables"]], [3, 2, 1])
        self.assertEqual(document["tables"][0]["rows"][0]["A L2"], "2.621e-08")
        self.assertEqual(document["tables"][2]["rows"][-1]["U H1 rate"], "1.00")
        recorded = json.loads(
            (path.parent / "convergence.json").read_text(encoding="utf-8")
        )
        self.assertEqual(recorded["provenance"]["sha256"], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(recorded["tables"], document["tables"])


@unittest.skipIf(generate_mms_expressions.sp is None, "SymPy is not installed")
class SymbolicDerivativeTests(unittest.TestCase):
    def test_steady_uniform_flow_requires_positive_friction_forcing(self):
        # With no time or spatial derivatives, the only momentum balance is
        # imposed source = viscous loss. For A=2, U=3, rho=5, mu=7, xi=0,
        # the required forcing is 8.4*pi and must act against the loss.
        profile = generate_mms_expressions.MMSProfile("uniform", "2", "3")
        result = generate_mms_expressions.derive_profile(profile)
        sp = generate_mms_expressions.sp
        source = sp.sympify(result["momentum_source"].replace("^", "**"),
                            locals={"PI": sp.pi})
        value = source.subs({"rho": 5, "mu": 7, "xi": 0})
        self.assertEqual(sp.simplify(value - sp.Rational(42, 5) * sp.pi), 0)
        self.assertEqual(result["mass_source"], "0")

    def test_sources_are_exact_symbolic_derivatives(self):
        profile = generate_mms_expressions.PROFILES["p2"]
        result = generate_mms_expressions.derive_profile(profile)
        x, t = generate_mms_expressions.sp.symbols("x t")
        # Re-parse parser spelling only for the independently checked fields;
        # source and velocity derivatives must not be finite-difference estimates.
        area = generate_mms_expressions.sp.sympify(profile.area.replace("PI", "pi"), locals={"x": x, "t": t, "pi": generate_mms_expressions.sp.pi})
        velocity = generate_mms_expressions.sp.sympify(profile.velocity.replace("PI", "pi"), locals={"x": x, "t": t, "pi": generate_mms_expressions.sp.pi})
        expected_area_t = generate_mms_expressions._parser(generate_mms_expressions.sp.diff(area, t))
        expected_velocity_x = generate_mms_expressions._parser(generate_mms_expressions.sp.diff(velocity, x))
        self.assertEqual(result["area_t"], expected_area_t)
        self.assertEqual(result["velocity_x"], expected_velocity_x)
        self.assertIn(";", result["rhs_expression"])


if __name__ == "__main__":
    unittest.main()
