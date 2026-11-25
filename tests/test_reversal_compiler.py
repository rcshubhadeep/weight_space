import unittest

import sympy as sp

from examples.convolution_child import rows
from examples.reversal_compiler import words


class ReversalTheoremTests(unittest.TestCase):
    def test_exact_fit_matrix_factors_in_word_basis(self):
        data = list(rows("fit"))
        vocab = sorted({w for s,_ in data for w in words(s)})
        basis = sp.eye(len(vocab))
        def unit(w):
            return basis[:,vocab.index(w)]
        observed = sp.zeros(len(vocab))
        for text,y in data:
            seq = words(text)
            for a,b in zip(seq,seq[1:]):
                observed[vocab.index(a),vocab.index(b)] += y
        roles = 2*unit("received")-unit("expected")-unit("expecting")
        polarity = sum((unit(p)-unit(n) for p in ("excellent","good","great")
                        for n in ("bad","poor","terrible")),sp.zeros(len(vocab),1))
        self.assertEqual(observed,roles*polarity.T)
        self.assertEqual(observed.rank(),1)

    def test_unequal_followers_leave_an_extra_term(self):
        # Role/sentiment entries for preceding words, swapped words and followers.
        u,v,x,y = sp.symbols("u v x y")
        ra,rb,sa,sb = sp.symbols("ra rb sa sb")
        difference = (u*sa+ra*x+v*sb+rb*y)-(u*sb+rb*x+v*sa+ra*y)
        self.assertEqual(sp.expand(difference-(u-v)*(sa-sb)-(ra-rb)*(x-y)),0)

