# Research direction: weight-space programming (first draft)

## North star

Can we build a programming environment that lets people specify, inspect, edit, compose, and reuse neural network behaviors, with evidence that intended changes work and other behaviors are preserved?

Weight-space programming resembles program synthesis: a specification describes desired behavior, and a system constructs or modifies weights to satisfy it. Unlike ordinary source code, learned representations may be distributed and hard to interpret, so evaluating edits and their side effects is central.

## Questions

1. What primitives let a person describe a behavior, inspect a network, constrain edits, and evaluate results? Can direct edits, linear programming, and other methods share an interface?
2. How can components from different networks be reused or composed? Matching tensor shapes alone does not ensure their internal representations are compatible.
3. Can an agent select, create, evaluate, improve, and store small networks for recurring user tasks, while retaining specifications and evidence of their behavior?

Start with a Python library to discover the useful abstractions. Consider a dedicated language only if experience shows that it makes specifications or workflows clearer.

## First investigation

Build a small CPU-friendly workbench: specify target and preservation examples, inspect a network, edit selected weights, and compare outcomes. Begin with a deliberately flawed paddle policy, then move toward trained networks where representations are not hand-designed. Compare future automated edits against simple baselines and evaluate on held-out cases.

## References

- [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
- [Linear Programming for Weight-Space Programming](https://github.com/nathanrs/weight-space-lp)
- [Language Models Can Rewrite Their Own Weights](https://arxiv.org/abs/2611.21885)

These are research inspirations, not evidence that every proposed direction is novel or already achievable.
