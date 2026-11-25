"""Exact sign checks for a linear term plus a scaled square root of a quadratic."""

from dataclasses import dataclass
from fractions import Fraction


def vector(values):
    """Represent a vector of finite binary floats with one power-of-two denominator."""
    ratios = [value.as_integer_ratio() for value in values]
    denominator = max(divisor for _, divisor in ratios)
    return [numerator * (denominator // divisor) for numerator, divisor in ratios], denominator


def dot(a, b):
    return Fraction(sum(x * y for x, y in zip(a[0], b[0], strict=True)), a[1] * b[1])


def multiply(a, b):
    return [x * y for x, y in zip(a[0], b[0], strict=True)], a[1] * b[1]


def subtract(a, b):
    denominator = max(a[1], b[1])
    left, right = denominator // a[1], denominator // b[1]
    return [x * left - y * right for x, y in zip(a[0], b[0], strict=True)], denominator


def quadratic_minimum(a, b, c, left, right):
    def value(t):
        return (a * t + b) * t + c

    if a > 0:
        vertex = -b / (2 * a)
        if left < vertex < right:
            return value(vertex)
    return min(value(left), value(right))


@dataclass(frozen=True)
class Decision:
    """Numerator R sqrt(C + D t + E t²) + U + V t, with a positive radicand."""

    R: Fraction
    C: Fraction
    D: Fraction
    E: Fraction
    U: Fraction
    V: Fraction

    def sign_at(self, t):
        linear = self.U + self.V * t
        if self.R == 0 or linear == 0 or (self.R > 0) == (linear > 0):
            chosen = linear if self.R == 0 else self.R
            return (chosen > 0) - (chosen < 0)
        squared = self.R**2 * (self.C + t * (self.D + self.E * t)) - linear**2
        return ((squared > 0) - (squared < 0)) * (1 if self.R > 0 else -1)

    def stable_through(self, end):
        """Whether the sign remains strictly equal to the initial sign on [0, end]."""
        if end < 0:
            raise ValueError("The strength interval must be nonnegative")
        zero = Fraction(0)
        sign = self.sign_at(zero)
        if sign == 0:
            return False
        root_term, constant, slope = sign * self.R, sign * self.U, sign * self.V
        at_end = constant + slope * end
        if root_term == 0:
            return min(constant, at_end) > 0

        a = root_term**2 * self.E - slope**2
        b = root_term**2 * self.D - 2 * constant * slope
        c = root_term**2 * self.C - constant**2
        if root_term < 0:
            return min(constant, at_end) > 0 and quadratic_minimum(-a, -b, -c, zero, end) > 0
        if min(constant, at_end) >= 0:
            return True
        left = -constant / slope if constant > 0 else zero
        right = -constant / slope if at_end > 0 else end
        return quadratic_minimum(a, b, c, left, right) > 0

    def first_boundary_bracket(self, horizon, steps=18):
        """Return exact rational bounds on the first tie, or None through the horizon."""
        lower, upper = Fraction(0), Fraction(horizon)
        if self.stable_through(upper):
            return None
        if self.sign_at(lower) == 0:
            return lower, lower
        for _ in range(steps):
            midpoint = (lower + upper) / 2
            if self.stable_through(midpoint):
                lower = midpoint
            else:
                upper = midpoint
        return lower, upper


def build_decisions(experiment, capture, own_key, cross_key):
    """Assemble exact coefficients from the captured tensors' binary values."""
    keys, residual, raw = [part.detach().cpu().tolist() for part in capture]
    own, cross = [vector(key.detach().cpu().tolist()) for key in (own_key, cross_key)]
    output = vector(experiment.value.detach().cpu().tolist())
    rows = [vector(row) for row in experiment.answer_rows.detach().cpu().tolist()]
    final_gain = vector(experiment.final_norm.weight.detach().cpu().tolist())
    post_gain = vector(experiment.post_norm.weight.detach().cpu().tolist())
    readout = multiply(subtract(rows[1], rows[0]), final_gain)
    scaled_readout = multiply(readout, post_gain)
    output_projection = dot(output, scaled_readout)
    output_norm = dot(output, output)
    dimension = len(output[0])
    epsilon = Fraction(experiment.post_norm.variance_epsilon)
    if epsilon <= 0 or experiment.final_norm.variance_epsilon <= 0:
        raise ValueError("The sign reduction requires positive RMS epsilons")
    decisions = []
    for x, r, z in zip(keys, residual, raw, strict=True):
        x, residual_row, raw_row = vector(x), vector(r), vector(z)
        own_scale, cross_scale = dot(x, own), dot(x, cross)
        cross_product = dot(raw_row, output) + own_scale * output_norm
        squared_norm = dot(raw_row, raw_row) + own_scale * (2 * cross_product - own_scale * output_norm)
        decisions.append(Decision(
            dot(residual_row, readout),
            squared_norm / dimension + epsilon,
            2 * cross_scale * cross_product / dimension,
            cross_scale**2 * output_norm / dimension,
            dot(raw_row, scaled_readout) + own_scale * output_projection,
            cross_scale * output_projection,
        ))
    return decisions
