# TeneT VUMPS Step Runners

`run_tenetc.jl` emits `TENETC_VUMPS_STEP` rows for the TeneT.c/FastTeneT path.
`run_tenet_ipeps_unified.jl` emits `TENET_IPEPS_VUMPS_STEP` rows for the pinned
TeneT.jl iPEPS-unified baseline.

Both runners initialize once per `chi`, perform warmup steps, then time a single
warmed VUMPS step over repeated measurements. The comparison script writes a TSV
with step medians and same-backend runtime ratios.

VUMPS convergence error is intentionally absent from this benchmark schema.
