# Generated Benchmark Tables

## Completed-Baseline Comparison

| chi | TeneT.jl master median (s) | TeneT.c median (s) | speedup | master error | TeneT.c error | status |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 64 | 40.995462 | 2.188001 | 18.74x | 1.53e-05 | 1.33e-05 | measured |
| 128 | not measured | 2.948597 | n/a | n/a | 6.68e-06 | timeout |
| 256 | not measured | 4.467242 | n/a | n/a | 3.67e-06 | timeout |

## Native Scaling

| chi | TeneT.c median (s) | p25 (s) | p75 (s) | TeneT.c error |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 2.188001 | 2.187011 | 2.189780 | 1.33e-05 |
| 128 | 2.948597 | 2.946695 | 2.953534 | 6.68e-06 |
| 256 | 4.467242 | 4.464754 | 4.469334 | 3.67e-06 |
