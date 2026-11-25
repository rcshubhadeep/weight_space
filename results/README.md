# Experiment evidence

[November 25 snapshot](2026-11-25/manifest.json) indexes the frozen reports on edits, composition, standalone children, and reversal recovery. The [experimental record](../docs/EXPERIMENTS.md) connects the research questions to results and limitations. Generated checkpoints and datasets are obtained using the [replication steps](../docs/REPLICATION.md).

```sh
python scripts/verify_evidence.py
```

The verifier checks canonical report hashes, key counts, audited predictions, and disjoint-mask witnesses. It can also verify downloaded corpus and checkpoint files, the original direct-write patch, and regenerated replay metrics. The snapshot substitutes a public model identifier and revision for two machine-specific checkpoint paths. Hashes establish consistency with the snapshot. Experimental replication establishes whether an observed result recurs.
