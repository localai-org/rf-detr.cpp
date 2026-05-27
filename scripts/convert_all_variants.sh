#!/usr/bin/env bash
# scripts/convert_all_variants.sh -- converts all 5 RF-DETR detection variants
# to GGUF F32. Each conversion downloads ~30-130 MB on first run (cached at
# ~/.roboflow/models/) and takes ~10-30s to load + a few seconds to write.
#
# Skips per-variant if the output GGUF already exists, so re-running this
# after a partial run will only convert the missing ones.

set -euo pipefail
cd "$(dirname "$0")/.."

VARIANTS=(nano small base medium large)
mkdir -p models
for v in "${VARIANTS[@]}"; do
    out="models/rfdetr-${v}-f32.gguf"
    if [[ -f "$out" ]]; then
        echo "skipping $v (file exists: $out)"
        continue
    fi
    echo "=== converting $v ==="
    .venv/bin/python scripts/convert_rfdetr_to_gguf.py \
        --variant "$v" --dtype f32 --output "$out"
    ls -lh "$out"
done

echo ""
echo "all variants converted. Summary:"
ls -lh models/rfdetr-*-f32.gguf
