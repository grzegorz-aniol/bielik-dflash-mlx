#!/usr/bin/env python
"""
Benchmark Bielik 11B v3 on MLX (AR or DFlash) with variable prompt/generation depths.

Usage:
  python scripts/benchmark_sweep.py --mode ar   --prompt-depths 512 1024 --gen-depths 128 256
  python scripts/benchmark_sweep.py --mode dflash --prompt-depths 512 --gen-depths 128
"""

import argparse
import csv
import json
import time
from pathlib import Path

from huggingface_hub import snapshot_download

import mlx.core as mx
import mlx.nn as nn

_BASE_TEXT = (
    "Przemyslaw jest wysokim, inteligentnym mezczyzna, "
    "ktory lubi chodzic na dlugie spacery po parku i czytac "
    "ksiazki o tematyce historycznej. "
)


def make_prompt_ids(tokenizer, prompt_len):
    base_ids = tokenizer.encode(_BASE_TEXT)
    if base_ids and tokenizer.bos_token_id is not None and base_ids[0] == tokenizer.bos_token_id:
        base_ids = base_ids[1:]
    if not base_ids:
        base_ids = [1]
    n = len(base_ids)
    repeats = (prompt_len // n) + 1
    return (base_ids * repeats)[:prompt_len]


def resolve_draft_path(path_or_id: str) -> Path:
    p = Path(path_or_id)
    if p.exists():
        return p
    return Path(snapshot_download(path_or_id))


def load_quantized_draft(path_or_id: str):
    from dflash.model_mlx import DFlashConfig, DFlashDraftModel

    path = resolve_draft_path(path_or_id)
    cfg = json.loads((path / "config.json").read_text())
    lt = tuple(cfg.get("layer_types") or ["full_attention"] * cfg["num_hidden_layers"])
    config = DFlashConfig(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"],
        max_position_embeddings=cfg["max_position_embeddings"],
        block_size=cfg["block_size"],
        target_layer_ids=tuple(cfg["dflash_config"]["target_layer_ids"]),
        num_target_layers=cfg["num_target_layers"],
        mask_token_id=cfg["dflash_config"]["mask_token_id"],
        rope_scaling=cfg.get("rope_scaling"),
        layer_types=lt,
        sliding_window=cfg.get("sliding_window"),
        final_logit_softcapping=cfg.get("final_logit_softcapping"),
    )
    draft = DFlashDraftModel(config)
    q = cfg.get("quantization", {})
    if q:
        nn.quantize(draft, group_size=q.get("group_size", 32), bits=q.get("bits", 8))
    weights = {
        k: v
        for f in sorted(path.glob("*.safetensors"))
        for k, v in mx.load(str(f)).items()
    }
    draft.load_weights(list(weights.items()))
    draft.eval()
    return draft


def run_ar(model, tokenizer, prompt_ids, gen_len, sampler):
    from mlx_lm.generate import stream_generate

    last = None
    for r in stream_generate(
        model, tokenizer, prompt_ids,
        max_tokens=gen_len, sampler=sampler,
    ):
        last = r
    return last


def run_dflash(model, draft, tokenizer, prompt_ids, gen_len, block_size, sampler):
    from dflash.model_mlx import stream_generate

    last = None
    for r in stream_generate(
        model, draft, tokenizer, prompt_ids,
        max_tokens=gen_len, block_size=block_size, sampler=sampler,
    ):
        last = r
    return last


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Bielik 11B v3 on MLX (AR or DFlash)",
    )
    parser.add_argument("--mode", choices=["ar", "dflash"], required=True)
    parser.add_argument("--prompt-depths", nargs="+", type=int, required=True,
                        help="Prompt token lengths to benchmark")
    parser.add_argument("--gen-depths", nargs="+", type=int, required=True,
                        help="Generation token lengths to benchmark")
    parser.add_argument("--trials", type=int, default=1,
                        help="Trials per combo (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results as CSV")
    parser.add_argument("--block-size", type=int, default=16,
                        help="DFlash block size (default: 16)")
    parser.add_argument("--target-model", type=str, default="speakleash/Bielik-11B-v3.0-Instruct-MLX-8bit",
                        help="Target model (HF repo ID or path, default: speakleash/Bielik-11B-v3.0-Instruct-MLX-8bit)")
    parser.add_argument("--draft-model", type=str, default=None,
                        help="Path or HF repo ID of quantized DFlash draft model (required for --mode dflash)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-trial progress")
    args = parser.parse_args()

    if args.mode == "dflash" and not args.draft_model:
        parser.error("--draft-model is required when --mode dflash")

    mx.random.seed(args.seed)

    from mlx_lm import load as mlx_lm_load

    print(f"[LOAD] {args.target_model} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    model, tokenizer = mlx_lm_load(args.target_model)
    mx.eval(model.parameters())
    print(f"{time.perf_counter() - t0:.1f}s")

    tokenizer._eos_token_ids = {}

    draft = None
    if args.mode == "dflash":
        print(f"[LOAD] {args.draft_model} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        draft = load_quantized_draft(args.draft_model)
        print(f"{time.perf_counter() - t0:.1f}s")

    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=0.0)

    results = []

    for prompt_len in args.prompt_depths:
        prompt_ids = make_prompt_ids(tokenizer, prompt_len)
        actual_len = len(prompt_ids)
        if actual_len != prompt_len:
            print(f"  [WARN] prompt_len={prompt_len} -> actual={actual_len}", flush=True)

        for gen_len in args.gen_depths:
            label = f"p={prompt_len} g={gen_len}"

            if args.verbose:
                print(f"\n  [{label}]   ", end="", flush=True)

            best_tps = 0.0
            peak_mem = 0.0

            for trial in range(args.trials):
                if args.mode == "ar":
                    r = run_ar(model, tokenizer, prompt_ids, gen_len, sampler)
                else:
                    r = run_dflash(model, draft, tokenizer, prompt_ids, gen_len,
                                   args.block_size, sampler)
                mx.clear_cache()

                tps = r.generation_tps if r else 0.0
                mem = mx.get_peak_memory() / 1e9 if r else 0.0

                if tps > best_tps:
                    best_tps = tps
                    peak_mem = mem

                if args.verbose:
                    print(f"{trial+1}:{tps:.1f}", end="  ", flush=True)

            if args.verbose:
                print(f"best={best_tps:.1f} tok/s  mem={peak_mem:.1f} GB")

            results.append((prompt_len, gen_len, best_tps, peak_mem))

    if args.output:
        with open(args.output, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["prompt_len", "gen_len", "generation_tps", "peak_memory_gb"])
            writer.writerows(results)

    print(f"\n{'Prompt':>6} {'Gen':>6} {'tok/s':>8} {'Mem(GB)':>8}")
    print("-" * 32)
    for prompt_len, gen_len, tps, mem in results:
        print(f"{prompt_len:>6} {gen_len:>6} {tps:>8.1f} {mem:>8.1f}")


if __name__ == "__main__":
    main()
