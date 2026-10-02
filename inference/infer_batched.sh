#!/bin/bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# : "${MODEL_PATH:-weights/ar-ckpt-giter057K-ep99-iter571-statedict.pth}"
# : "${EVAL_DATA_PATH:-/workspace/data/eval_namvis/rendered_objaverse8_wdepth_pitch30}"
# : "${VAE_PATH:-weights/infinity_vae_d32reg.pth}"

MODEL_PATH="${MODEL_PATH:-weights/namvis_1b.pth}"
VAE_PATH="${VAE_PATH:-weights/infinity_vae_d32reg.pth}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-/workspace/namvis/data_eval/rendered_objaverse8_wdepth_pitch30/}"

export model="1b"
RESULTS_ROOT="${RESULTS_ROOT:-inference_results_namvis256}"
GRID_ROOT="${GRID_ROOT:-inference_grid_namvis256}"

# Define your maximum values
MAX_VIEWS_SRC=3
MAX_VIEWS_TGT=3

# SRC_MASTER_INDICES=(0 3 7)
# TGT_MASTER_INDICES=(1 2 4 5 6)

SRC_MASTER_INDICES=(0 3 7)
TGT_MASTER_INDICES=(1 2 4 5 6)

SRC_OFFSET=2

# Outer loop for source views
for (( src=1; src<=MAX_VIEWS_SRC; src++ )); do

    # Inner loop for target views
    for (( tgt=1; tgt<=MAX_VIEWS_TGT; tgt++ )); do
        
        echo "=================================================="
        echo "Running inference with N_views_src=${src} and N_views_tgt=${tgt}"
        echo "=================================================="

        #--------------Dynamically set the output directory based on current loop variables

        current_objaverse_out="${RESULTS_ROOT}/src${src}_tgt${tgt}/objaverse"
        grid_objaverse_out="${GRID_ROOT}/src${src}_tgt${tgt}/objaverse"
        
        current_gso_out="${RESULTS_ROOT}/src${src}_tgt${tgt}/gso"
        grid_gso_out="${GRID_ROOT}/src${src}_tgt${tgt}/gso"

        current_oo3d_out="${RESULTS_ROOT}/src${src}_tgt${tgt}/oo3d"
        grid_oo3d_out="${GRID_ROOT}/src${src}_tgt${tgt}/oo3d"

        current_nerf_out="${RESULTS_ROOT}/src${src}_tgt${tgt}/nerf"
        grid_nerf_out="${GRID_ROOT}/src${src}_tgt${tgt}/nerf"

        python3 inference/infer_ext.py \
            --data_path="${EVAL_DATA_PATH}" \
            --model_path="${MODEL_PATH}" \
            --vae_path="${VAE_PATH}" \
            --use_prope=1 \
            --pn="0.06M" \
            --N_views_src="${src}" \
            --N_views_tgt="${tgt}" \
            --out_dir="${current_objaverse_out}" \
            --dataset="objaverse" \
            --src_offset="${SRC_OFFSET}" \
            --tgt_offset="$((SRC_OFFSET + MAX_VIEWS_SRC))" \
            --model="${model}" \
            --cos=0 \
            --src_indices "${SRC_MASTER_INDICES[@]}" \
            --tgt_indices "${TGT_MASTER_INDICES[@]}" \
            --grid_out_dir="${grid_objaverse_out}"

        # python3 inference/infer_ext.py \
        #     --data_path="/workspace/data/eval_namvis/rendered_gso8_wdepth_pitch30" \
        #     --model_path="${MODEL_PATH}" \
        #     --pn="0.06M" \
        #     --N_views_src="${src}" \
        #     --N_views_tgt="${tgt}" \
        #     --out_dir="${current_gso_out}" \
        #     --dataset="objaverse" \
        #     --src_offset="${SRC_OFFSET}" \
        #     --tgt_offset="$((SRC_OFFSET + MAX_VIEWS_SRC))" \
        #     --model="${model}" \
        #     --cos=0 \
        #     --src_indices "${SRC_MASTER_INDICES[@]}" \
        #     --tgt_indices "${TGT_MASTER_INDICES[@]}" \
        #     --grid_out_dir="${grid_gso_out}"

        # python3 inference/infer_ext.py \
        #     --data_path="/workspace/data/eval_namvis/rendered_oo3d8_wdepth_pitch30" \
        #     --model_path="${MODEL_PATH}" \
        #     --pn="0.06M" \
        #     --N_views_src="${src}" \
        #     --N_views_tgt="${tgt}" \
        #     --out_dir="${current_oo3d_out}" \
        #     --dataset="objaverse" \
        #     --src_offset="${SRC_OFFSET}" \
        #     --tgt_offset="$((SRC_OFFSET + MAX_VIEWS_SRC))" \
        #     --model="${model}" \
        #     --cos=0 \
        #     --src_indices "${SRC_MASTER_INDICES[@]}" \
        #     --tgt_indices "${TGT_MASTER_INDICES[@]}" \
        #     --grid_out_dir="${grid_oo3d_out}"

        # Calculate metrics for this specific run
        echo "Calculating metrics for src=${src}, tgt=${tgt}..."
        
        echo "${current_objaverse_out}"
        python3 inference/calc_metric.py --root="${current_objaverse_out}"
        
        # echo "${current_gso_out}"
        # python3 inference/calc_metric.py --root="${current_gso_out}"
        
        # echo "${current_oo3d_out}"
        # python3 inference/calc_metric.py --root="${current_oo3d_out}"


    done
done

echo "All view combinations completed!"
