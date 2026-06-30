# Benchmark Results

This directory stores compact benchmark summaries used by README figures.

Raw jobctl artifacts should normally stay in the job run directory or under
`/tmp`. Commit only compact CSV/TSV summaries, host metadata, and generated
figures needed for README claims.

`metadata.toml` records the source and status for each committed artifact,
including timeout/not-measured baselines. Do not promote a row to a headline
claim unless the corresponding raw summary, host metadata, and source run ID are
present here.

Current GPU artifacts:

- `tenet_ipeps_h100.tsv`: official TeneT.jl `iPEPS-unified` real CUDA baseline
  against TeneT.c real CUDA, used for the headline GPU speedup figure.
- `tenetc_h100.tsv`: TeneT.jl `master` ComplexF64 timing audit against TeneT.c
  Float64, retained only as a scalar-mismatch audit.
- `tenetc_native_h100.tsv`: TeneT.c native real CUDA scaling, including larger
  `chi` values without implying a completed baseline.

Regenerate the Markdown table summary with:

```sh
python3 benchmarks/results/render_release_tables.py
```
