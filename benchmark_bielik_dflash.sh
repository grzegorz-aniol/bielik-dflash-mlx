#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/benchmark_sweep.py"

MODE="ar"
PROMPT_DEPTHS="512"
GEN_DEPTHS="128"
TRIALS=1
SEED=42
BLOCK_SIZE=16
TARGET="speakleash/Bielik-11B-v3.0-Instruct-MLX-8bit"
DRAFT=""
OUTPUT=""
VERBOSE=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [MODE] [OPTIONS]

Modes:
  --ar-only        Run AR benchmark only (default)
  --dflash-only    Run DFlash benchmark only
  --compare        Run AR then DFlash, print comparison

Options:
  --prompt-depths "P1 P2 ..."    Prompt token lengths (default: "$PROMPT_DEPTHS")
  --gen-depths "G1 G2 ..."       Generation token lengths (default: "$GEN_DEPTHS")
  --trials N                     Trials per combo (default: $TRIALS)
  --seed N                       Random seed (default: $SEED)
  --block-size N                 DFlash block size (default: $BLOCK_SIZE)
  --target-model ID              Target model (default: $TARGET)
  --draft-model PATH             Quantized draft model path or HF ID (required for dflash/compare)
  --output FILE                  Save AR/DFlash results as CSV
  --verbose                      Print per-trial progress
  -h, --help                     Show this help

Examples:
  $(basename "$0") --ar-only
  $(basename "$0") --compare --prompt-depths "512 1024 2048" --gen-depths "128 256 512"
  $(basename "$0") --dflash-only --prompt-depths "1024" --gen-depths "256" --verbose
EOF
    exit 0
}

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ar-only) MODE="ar"; shift ;;
        --dflash-only) MODE="dflash"; shift ;;
        --compare) MODE="compare"; shift ;;
        --prompt-depths) PROMPT_DEPTHS="$2"; shift 2 ;;
        --gen-depths) GEN_DEPTHS="$2"; shift 2 ;;
        --trials) TRIALS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --block-size) BLOCK_SIZE="$2"; shift 2 ;;
        --target-model) TARGET="$2"; shift 2 ;;
        --draft-model) DRAFT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --verbose) VERBOSE="--verbose"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

prompt_list() { echo "$PROMPT_DEPTHS" | tr ' ' ','; }
gen_list()    { echo "$GEN_DEPTHS" | tr ' ' ','; }

echo "=== Bielik 11B v3 MLX Benchmark ==="
echo "  mode:      $MODE"
echo "  prompts:   $(prompt_list)"
echo "  gen:       $(gen_list)"
echo "  trials:    $TRIALS"
echo "  seed:      $SEED"
echo "  block:     $BLOCK_SIZE"
echo ""

run_bench() {
    local mode="$1"
    local label="$2"
    local out_flag=""
    local extra_flag=""

    if [[ -n "$OUTPUT" ]]; then
        local base="${OUTPUT%.csv}"
        out_flag="--output ${base}_${mode}.csv"
    fi
    if [[ "$mode" == "dflash" && -n "$DRAFT" ]]; then
        extra_flag="--draft-model $DRAFT"
    fi

    echo "--- $label ---"
    uv run python "$PY_SCRIPT" \
        --mode "$mode" \
        --target-model "$TARGET" \
        --prompt-depths $PROMPT_DEPTHS \
        --gen-depths $GEN_DEPTHS \
        --trials "$TRIALS" \
        --seed "$SEED" \
        --block-size "$BLOCK_SIZE" \
        $VERBOSE \
        $out_flag \
        $extra_flag
    echo ""
}

case "$MODE" in
    ar)
        run_bench "ar" "AR (no draft)"
        ;;
    dflash)
        if [[ -z "$DRAFT" ]]; then
            echo "ERROR: --draft-model is required for --dflash-only mode" >&2
            exit 1
        fi
        run_bench "dflash" "DFlash (block_size=$BLOCK_SIZE)"
        ;;
    compare)
        if [[ -z "$DRAFT" ]]; then
            echo "ERROR: --draft-model is required for --compare mode" >&2
            exit 1
        fi
        TMP_AR=$(mktemp /tmp/bench_ar_XXXXX.csv)
        TMP_DF=$(mktemp /tmp/bench_df_XXXXX.csv)

        echo "=== PASS 1: AR (no draft) ==="
        uv run python "$PY_SCRIPT" \
            --mode ar \
            --target-model "$TARGET" \
            --prompt-depths $PROMPT_DEPTHS \
            --gen-depths $GEN_DEPTHS \
            --trials "$TRIALS" \
            --seed "$SEED" \
            --block-size "$BLOCK_SIZE" \
            --output "$TMP_AR" \
            $VERBOSE
        echo ""

        echo "=== PASS 2: DFlash (block_size=$BLOCK_SIZE) ==="
        uv run python "$PY_SCRIPT" \
            --mode dflash \
            --target-model "$TARGET" \
            --prompt-depths $PROMPT_DEPTHS \
            --gen-depths $GEN_DEPTHS \
            --trials "$TRIALS" \
            --seed "$SEED" \
            --block-size "$BLOCK_SIZE" \
            --draft-model "$DRAFT" \
            --output "$TMP_DF" \
            $VERBOSE
        echo ""

        echo "=== COMPARISON ==="
        printf "%-8s %-8s %-10s %-10s %-8s\n" "Prompt" "Gen" "AR tok/s" "DF tok/s" "Speedup"
        printf "%s\n" "------------------------------------------------"
        paste -d, \
            <(tail -n +2 "$TMP_AR" | sort -t, -k1,1n -k2,2n) \
            <(tail -n +2 "$TMP_DF" | sort -t, -k1,1n -k2,2n) \
        | while IFS=, read -r p1 g1 tps1 mem1 p2 g2 tps2 mem2; do
            if [[ -z "$tps1" || -z "$tps2" ]]; then
                continue
            fi
            speedup=$(echo "scale=1; ($tps2 / $tps1 - 1) * 100" | bc -l 2>/dev/null || echo "0.0")
            printf "%-8s %-8s %-10.1f %-10.1f %+7.0f%%\n" "$p1" "$g1" "$tps1" "$tps2" "$speedup"
        done
        printf "%s\n" "------------------------------------------------"

        rm -f "$TMP_AR" "$TMP_DF"

        if [[ -n "$OUTPUT" ]]; then
            {
                echo "prompt_len,gen_len,ar_tps,dflash_tps,speedup_pct"
                paste -d, \
                    <(tail -n +2 "$TMP_AR" | sort -t, -k1,1n -k2,2n) \
                    <(tail -n +2 "$TMP_DF" | sort -t, -k1,1n -k2,2n) \
                | while IFS=, read -r p1 g1 tps1 mem1 p2 g2 tps2 mem2; do
                    if [[ -z "$tps1" || -z "$tps2" ]]; then
                        continue
                    fi
                    speedup=$(echo "scale=1; ($tps2 / $tps1 - 1) * 100" | bc -l 2>/dev/null || echo "0.0")
                    echo "$p1,$g1,$tps1,$tps2,$speedup"
                done
            } > "$OUTPUT"
            echo "Combined CSV saved to: $OUTPUT"
        fi
        ;;
esac
