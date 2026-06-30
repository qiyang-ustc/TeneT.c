# Reproducing TeneT.c Results

Use jobctl from the repository root:

```bash
bash benchmarks/run_release_suite.sh
```

The CPU job runs on Oblix with `--cpus 4 --mem 4G --time 02:00:00`.
The GPU job runs on Snellius H100 with `--partition gpu_h100 --gres gpu:h100:1
--cpus 16 --mem 180G --time 01:30:00`.

Both jobs use `chi = 32,64,96,128,160,192,224,256`, 3 warmup steps, 11 timed
repeats, `TENET_IPEPS_METHOD=krylovkit`, and `TENET_IPEPS_MODE=general`.
