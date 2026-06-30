# Release Results

Official result artifacts are produced by jobctl. The Snellius H100 and Oblix
CPU artifacts are present.

Expected artifacts:

- `vumps_step_gpu_snellius_h100.tsv`: completed, `run-0159d0af7c1a`.
- `vumps_step_cpu_oblix.tsv`: completed, `run-4889332039c9`.

Each TSV must contain 8 `chi` rows, same-backend `Float64` timing, and no VUMPS
error column.
