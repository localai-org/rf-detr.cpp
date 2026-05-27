#!/usr/bin/env bash
# scripts/build_all_quants.sh — materialize the full RF-DETR.cpp model matrix.
#
# For each detection + segmentation variant, ensures we have F32, F16, Q8_0,
# and Q4_K GGUF files in models/.  F32 source files must already exist
# (produced by scripts/convert_rfdetr_to_gguf.py); the other formats are
# derived in-place by the C++ quantizer (build/bin/rfdetr-cli quantize).
#
# Idempotent: existing output files are skipped.
#
# Usage:
#   scripts/build_all_quants.sh
#
# Coverage matrix (32 models):
#   detection × {nano, small, base, medium, large} × {F32, F16, Q8_0, Q4_K}
#   segment   × {seg-nano, seg-small, seg-medium}   × {F32, F16, Q8_0, Q4_K}

set -euo pipefail
cd "$(dirname "$0")/.."

CLI="build/bin/rfdetr-cli"
if [[ ! -x "$CLI" ]]; then
    echo "ERROR: $CLI not found or not executable. Run 'cmake --build build'." >&2
    exit 1
fi

DET_VARIANTS=(nano small base medium large)
SEG_VARIANTS=(seg-nano seg-small seg-medium)
QUANTS=(f16 q8_0 q4_K)

built=0
skipped=0
failed=0
failures=()

for v in "${DET_VARIANTS[@]}" "${SEG_VARIANTS[@]}"; do
    src="models/rfdetr-${v}-f32.gguf"
    if [[ ! -f "$src" ]]; then
        echo "WARN: $src missing — run:"
        echo "      scripts/convert_rfdetr_to_gguf.py --variant $v --output $src"
        failed=$((failed + 1))
        failures+=("${v}/<f32-source-missing>")
        continue
    fi
    for q in "${QUANTS[@]}"; do
        out="models/rfdetr-${v}-${q}.gguf"
        if [[ -f "$out" ]]; then
            echo "  skip   ${out} (exists)"
            skipped=$((skipped + 1))
            continue
        fi
        echo "=== ${v} → ${q} ==="
        if "$CLI" quantize "$src" "$out" "$q"; then
            ls -lh "$out"
            built=$((built + 1))
        else
            echo "FAIL  ${v} → ${q}" >&2
            failed=$((failed + 1))
            failures+=("${v}/${q}")
        fi
    done
done

echo
echo "=== Summary ==="
echo "  built:   $built"
echo "  skipped: $skipped"
echo "  failed:  $failed"
if (( failed > 0 )); then
    echo "  failures:"
    for f in "${failures[@]}"; do
        echo "    - $f"
    done
    exit 1
fi
