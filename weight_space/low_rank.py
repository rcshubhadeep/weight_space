"""Linear algebra used by the protected rank-one experiments."""

import torch


def fit_key(keys, shifts, protected=None):
    """Fit a ridge-regularized key orthogonal to the protected input span."""
    if protected is not None:
        basis = torch.linalg.qr(protected.T, mode="reduced").Q
        keys = keys - (keys @ basis) @ basis.T
    gram = keys @ keys.T
    ridge = 0.02 * gram.diag().mean().clamp_min(1e-6)
    system = gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    return keys.T @ torch.linalg.solve(system, shifts)


def add_rank_one_branch(keys, output, value, key, scale):
    """Add the rank-one contribution in float32 before the caller rounds it."""
    return output.float() + (scale * (keys.float() @ key)).unsqueeze(-1) * value
