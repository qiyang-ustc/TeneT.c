# Release JobFiles

Official release jobs:

- `oblix_tenet_vumps_cpu.jobfile.yaml`: CPU run on Oblix `lerner`, requested
  as `--partition lerner --cpus 4 --mem 4G --time 02:00:00`.
- `snellius_tenet_vumps_h100.jobfile.yaml`: GPU run on Snellius H100, requested
  as `--partition gpu_h100 --gres gpu:h100:1 --cpus 16 --mem 180G`.

Both jobs run the same `chi` sweep and write their comparison TSV into the
jobctl result directory.
