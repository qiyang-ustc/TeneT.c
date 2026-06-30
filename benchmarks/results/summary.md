# Generated Benchmark Tables

## Real GPU Baseline

| chi | TeneT.jl branch | TeneT.jl eltype | TeneT.c eltype | TeneT.jl median (s) | TeneT.c median (s) | speedup | TeneT.jl VUMPS error | TeneT.jl abs delta f | TeneT.c error | run id |
| ---: | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 32 | iPEPS-unified | Float64 | Float64 | 16.987717 | 1.404854 | 12.09x | 2.02e-05 | 5.89e-08 | 3.36e-05 | run-64f60d92fdcb |
| 48 | iPEPS-unified | Float64 | Float64 | 17.306478 | 1.719294 | 10.07x | 6.76e-06 | 6.73e-08 | 2.75e-05 | run-1a5c19416816 |
| 64 | iPEPS-unified | Float64 | Float64 | 17.734378 | 2.109899 | 8.41x | 1.53e-05 | 5.55e-08 | 1.33e-05 | run-d607011acacb |
| 96 | iPEPS-unified | Float64 | Float64 | 17.928481 | 2.491735 | 7.20x | 1.22e-05 | 5.60e-08 | 7.17e-06 | run-279bfc36172c |
| 128 | iPEPS-unified | Float64 | Float64 | 18.124031 | 2.830716 | 6.40x | 5.32e-06 | 5.33e-08 | 6.68e-06 | run-c2af6466687a |

## Completed GPU Timing Audit

| chi | master eltype | TeneT.c eltype | TeneT.jl master GPU median (s) | TeneT.c GPU median (s) | ratio | master error | TeneT.c error | comparison status |
| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| 32 | ComplexF64 | Float64 | 39.827650 | 1.404854 | 28.35x | 2.03e-05 | 3.36e-05 | scalar_mismatch_audit_only |
| 48 | ComplexF64 | Float64 | 38.011892 | 1.719294 | 22.11x | 1.51e-05 | 2.75e-05 | scalar_mismatch_audit_only |
| 64 | ComplexF64 | Float64 | 44.800633 | 2.109899 | 21.23x | 1.28e-05 | 1.33e-05 | scalar_mismatch_audit_only |
| 96 | ComplexF64 | Float64 | 68.559429 | 2.491735 | 27.51x | 7.55e-06 | 7.17e-06 | scalar_mismatch_audit_only |
| 128 | ComplexF64 | Float64 | 422.869580 | 2.830716 | 149.39x | 6.15e-06 | 6.68e-06 | scalar_mismatch_audit_only |
| 192 | not measured | Float64 | not measured | 3.331687 | n/a | n/a | 4.23e-06 | not measured |
| 256 | not measured | Float64 | not measured | 4.234532 | n/a | n/a | 3.85e-06 | not measured |
| 384 | not measured | Float64 | not measured | 7.143338 | n/a | n/a | 2.81e-06 | not measured |

## Native GPU Scaling

| chi | backend | eltype | TeneT.c GPU median (s) | p25 (s) | p75 (s) | TeneT.c error |
| ---: | :--- | :--- | ---: | ---: | ---: | ---: |
| 32 | cuda | Float64 | 1.404854 | 1.404048 | 1.405367 | 3.36e-05 |
| 48 | cuda | Float64 | 1.719294 | 1.718513 | 1.719433 | 2.75e-05 |
| 64 | cuda | Float64 | 2.109899 | 2.109182 | 2.113542 | 1.33e-05 |
| 96 | cuda | Float64 | 2.491735 | 2.491423 | 2.492150 | 7.17e-06 |
| 128 | cuda | Float64 | 2.830716 | 2.830250 | 2.831308 | 6.68e-06 |
| 192 | cuda | Float64 | 3.331687 | 3.328526 | 3.333417 | 4.23e-06 |
| 256 | cuda | Float64 | 4.234532 | 4.230963 | 4.237986 | 3.85e-06 |
| 384 | cuda | Float64 | 7.143338 | 7.140215 | 7.143937 | 2.81e-06 |
