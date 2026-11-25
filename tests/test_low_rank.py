from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from examples.lfm_patch_composition import margin_envelope
from weight_space.low_rank import add_rank_one_branch, fit_key


class RankOneEvidenceTests(unittest.TestCase):
    def test_branch_matches_dense_update_without_mutating_input_weight(self):
        keys = torch.tensor([[1.0, -2.0, 3.0], [-3.0, 1.0, 2.0]])
        weight = torch.tensor([[2.0, 3.0, 4.0], [-1.0, 1.0, 3.0]])
        original = weight.clone()
        value = torch.tensor([2.0, -3.0])
        key = torch.tensor([1.0, 3.0, -2.0])
        scale = 0.25
        patched = add_rank_one_branch(keys, F.linear(keys, weight), value, key, scale)
        expected = F.linear(keys, weight + scale * torch.outer(value, key))
        torch.testing.assert_close(patched, expected)
        self.assertTrue(torch.equal(weight, original))

    def test_protected_solve_matches_primal_ridge_in_nullspace(self):
        keys = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 2.0]], dtype=torch.float64)
        protected = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
        shifts = torch.tensor([0.5, -1.0], dtype=torch.float64)
        key = fit_key(keys, shifts, protected)
        torch.testing.assert_close(protected @ key, torch.zeros(1, dtype=keys.dtype), atol=1e-12, rtol=0)

        orthogonal = torch.linalg.qr(protected.T, mode="complete").Q[:, 1:]
        reduced_keys = keys @ orthogonal
        ridge = 0.02 * reduced_keys.square().sum(dim=1).mean()
        system = reduced_keys.T @ reduced_keys + ridge * torch.eye(2, dtype=keys.dtype)
        expected = orthogonal @ torch.linalg.solve(system, reduced_keys.T @ shifts)
        torch.testing.assert_close(key, expected)
        self.assertLess(float(torch.linalg.vector_norm(keys @ key - shifts)), 0.05)

    def test_cross_protection_leaves_each_own_anchor_response_intact(self):
        a = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
        b = torch.tensor([[0.0, 1.0, 1.0, 0.0]], dtype=torch.float64)
        shift = torch.tensor([1.0], dtype=torch.float64)
        key_a = fit_key(a, shift, b)
        key_b = fit_key(b, -shift, a)
        combined = key_a + key_b
        torch.testing.assert_close(a @ combined, a @ key_a, atol=1e-12, rtol=0)
        torch.testing.assert_close(b @ combined, b @ key_b, atol=1e-12, rtol=0)
        self.assertGreater(float(a @ key_a), 0.9)
        self.assertLess(float(b @ key_b), -0.9)

    def test_rms_margin_bound_contains_the_exact_composed_margin(self):
        generator = torch.Generator().manual_seed(110)
        raw = torch.randn(24, 5, generator=generator)
        keys = torch.randn(24, 7, generator=generator)
        residual = torch.randn(24, 5, generator=generator)
        case = SimpleNamespace(
            value=F.normalize(torch.randn(5, generator=generator), dim=0),
            post_norm=SimpleNamespace(weight=torch.linspace(0.8, 1.2, 5), variance_epsilon=1e-5),
            final_norm=SimpleNamespace(weight=torch.linspace(0.9, 1.1, 5), variance_epsilon=1e-5),
            answer_rows=torch.randn(2, 5, generator=generator),
        )
        own = torch.randn(7, generator=generator) * 0.2
        cross = torch.randn(7, generator=generator) * 0.02
        _, margin, bound = margin_envelope(case, (keys, residual, raw), own, cross)

        def rms(x, norm):
            return x * (x.square().mean(dim=-1, keepdim=True) + norm.variance_epsilon).rsqrt() * norm.weight

        ffn = raw + (keys @ (own + cross)).unsqueeze(-1) * case.value
        hidden = rms(residual + rms(ffn, case.post_norm), case.final_norm)
        exact = hidden @ (case.answer_rows[1] - case.answer_rows[0])
        self.assertTrue(torch.all((exact - margin).abs() <= bound + 1e-5))
        self.assertGreater(int((margin.abs() > bound).sum()), 0)
