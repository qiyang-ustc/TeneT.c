# Benchmark Results

This directory stores compact benchmark summaries used by README figures.

Raw jobctl artifacts should normally stay in the job run directory or under
`/tmp`. Commit only compact CSV/TSV summaries, host metadata, and generated
figures needed for README claims.

`metadata.toml` records the source and status for each committed artifact,
including timeout/not-measured baselines. Do not promote a row to a headline
claim unless the corresponding raw summary, host metadata, and source run ID are
present here.

Regenerate the Markdown table summary with:

```sh
python3 benchmarks/results/render_release_tables.py
```
