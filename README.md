# bielik-dflash-mlx

Benchmark Bielik 11B v3 on Apple Silicon with AR vs DFlash speculative decoding.

```bash
uv sync
```

```bach
./benchmark_bielik_dflash.sh --compare \
    --target-model speakleash/Bielik-11B-v3.0-Instruct-MLX-8bit \
    --draft-model gangel/Bielik-11B-v3.0-DFlash-MLX-8bit \
    --prompt-depths "512 2048 4096" \
    --gen-depths "128 256 512" \
    --trials 1
```
