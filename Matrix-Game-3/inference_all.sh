#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="demo_images"
OUTPUT_DIR="./output"

for example_dir in "${DEMO_DIR}"/*/; do
    example_id="$(basename "${example_dir}")"
    image_path="${example_dir}image.png"
    prompt_path="${example_dir}prompt.txt"

    if [[ ! -f "${image_path}" || ! -f "${prompt_path}" ]]; then
        echo "Skipping ${example_id}: missing image.png or prompt.txt"
        continue
    fi

    prompt="$(cat "${prompt_path}")"

    echo "=========================================="
    echo "Generating example ${example_id}"
    echo "Prompt: ${prompt}"
    echo "=========================================="

    torchrun --nproc_per_node=8 generate.py \
             --size "704*1280" \
             --dit_fsdp --t5_fsdp \
             --ckpt_dir Matrix-Game-3.0 --fa_version 3 \
             --use_int8 --num_iterations 12 --num_inference_steps 3 \
             --image "${image_path}" \
             --prompt "${prompt}" \
             --save_name "${example_id}" --seed 42 --compile_vae \
             --lightvae_pruning_rate 0.5 --vae_type mg_lightvae --output_dir "${OUTPUT_DIR}"
done

echo "All demo examples processed."
