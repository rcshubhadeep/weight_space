from fractions import Fraction
from types import SimpleNamespace
import unittest

import sympy as sp
import torch

from weight_space.decision_boundary import Decision, build_decisions


def decision(*coefficients):
    return Decision(*map(Fraction, coefficients))


class DecisionBoundaryTests(unittest.TestCase):
    def test_both_endpoints_can_agree_while_the_interior_flips(self):
        curve = decision(1, "23/20", -1, 1, -1, 0)
        self.assertEqual([curve.sign_at(t) for t in (0, Fraction(1, 2), 1)], [1, -1, 1])
        self.assertFalse(curve.stable_through(1))
        lower, upper = curve.first_boundary_bracket(1)
        self.assertTrue(curve.stable_through(lower))
        self.assertFalse(curve.stable_through(upper))
        root = (1 - sp.sqrt(sp.Rational(2, 5))) / 2
        self.assertLessEqual(sp.Rational(lower.numerator, lower.denominator), root)
        self.assertGreaterEqual(sp.Rational(upper.numerator, upper.denominator), root)

    def test_tangent_and_spurious_squared_roots(self):
        tangent = decision(1, "5/4", -1, 1, -1, 0)
        self.assertEqual(tangent.sign_at(Fraction(1, 2)), 0)
        self.assertFalse(tangent.stable_through(1))
        spurious = decision(1, 1, 0, 1, 1, 0)
        self.assertIsNone(spurious.first_boundary_bracket(1))
        negative_radical = decision(-1, 1, 0, 1, 2, 0)
        self.assertTrue(negative_radical.stable_through(1))
        self.assertFalse(negative_radical.stable_through(2))
        linear = decision(0, 1, 0, 0, -1, 2)
        self.assertTrue(linear.stable_through(Fraction(1, 4)))
        self.assertFalse(linear.stable_through(Fraction(1, 2)))

    def test_captured_coefficients_agree_with_symbolic_two_normalization_model(self):
        capture = (torch.tensor([[1.0, 2.0]]), torch.tensor([[-1.0, 0.5]]), torch.tensor([[2.0, -1.0]]))
        model = SimpleNamespace(
            value=torch.tensor([1.0, -0.5]),
            answer_rows=torch.tensor([[0.5, 1.0], [1.5, 0.0]]),
            post_norm=SimpleNamespace(weight=torch.tensor([2.0, 1.0]), variance_epsilon=0.25),
            final_norm=SimpleNamespace(weight=torch.tensor([1.0, 2.0]), variance_epsilon=0.125),
        )
        actual, = build_decisions(model, capture, torch.tensor([0.5, 0.0]), torch.tensor([0.0, -1.0]))
        t = sp.Symbol("t", real=True)
        r = sp.Matrix([-1, sp.Rational(1, 2)])
        z = sp.Matrix([2, -1]) + (sp.Rational(1, 2) - 2 * t) * sp.Matrix([1, -sp.Rational(1, 2)])
        norm_squared = z.dot(z) / 2 + sp.Rational(1, 4)
        h = r + sp.diag(2, 1) * z / sp.sqrt(norm_squared)
        full_margin = sp.Matrix([1, -1]).dot(sp.diag(1, 2) * h) / sp.sqrt(h.dot(h) / 2 + sp.Rational(1, 8))
        expected = sp.simplify(sp.Matrix([1, -2]).dot(h) * sp.sqrt(norm_squared))
        actual_expr = actual.R * sp.sqrt(actual.C + actual.D * t + actual.E * t**2) + actual.U + actual.V * t
        self.assertEqual(sp.simplify(actual_expr - expected), 0)
        for strength in (0, Fraction(1, 4), Fraction(1, 2), 1):
            self.assertEqual(int(sp.sign(full_margin.subs(t, strength))), actual.sign_at(strength))
