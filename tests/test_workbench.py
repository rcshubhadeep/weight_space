import unittest

import torch
from torch import nn

from weight_space import Case, compare, evaluate, set_parameter


class WorkbenchTests(unittest.TestCase):
    def test_edit_and_compare_target_and_preservation(self):
        model = nn.Linear(2, 2)
        with torch.no_grad():
            model.weight[:] = torch.tensor([[-1.0, -1.0], [-1.0, 1.0]])
            model.bias.zero_()
        cases = [Case("left", (5.0, 2.0), 0, "target"),
                 Case("right", (5.0, 8.0), 1, "preserve")]
        before = evaluate(model, cases)
        edit = set_parameter(model, "weight", (0, 0), 1.0)
        summary = compare(before, evaluate(model, cases))
        self.assertEqual((edit["before"], edit["after"]), (-1.0, 1.0))
        self.assertEqual(summary["groups"]["target"]["before_passed"], 0)
        self.assertEqual(summary["groups"]["target"]["after_passed"], 1)
        self.assertEqual(summary["groups"]["preserve"]["after_passed"], 1)

    def test_invalid_edit_and_mismatched_cases(self):
        model = nn.Linear(2, 2)
        with self.assertRaises(IndexError):
            set_parameter(model, "weight", (0,), 1.0)
        with self.assertRaises(ValueError):
            compare(evaluate(model, [Case("a", (1, 2), 0, "x")]),
                    evaluate(model, [Case("b", (1, 2), 0, "x")]))


if __name__ == "__main__":
    unittest.main()
