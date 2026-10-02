#!/usr/bin/env bash
# Image-only adaptation of Infinity3D/inference_geo/video.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="${MODEL_PATH:-weights/ar-ckpt-giter057K-ep99-iter571-statedict.pth}"
VAE_PATH="${VAE_PATH:-weights/infinity_vae_d32reg.pth}"
# Set EVAL_DATA_PATH for a folder of scenes, or SCENE_PATH for one scene.
scene_args=()
if [[ -n "${SCENE_PATH:-}" ]]; then
  scene_args=(--scene_path="${SCENE_PATH}")
else
  EVAL_DATA_PATH="${EVAL_DATA_PATH:-/workspace/data/eval_namvis256/objaverse8_wdepth/}"
  scene_args=(--data_path="${EVAL_DATA_PATH}")
fi

# Match the three source views used by infer_batched.sh by default.
read -r -a source_indices <<< "${SOURCE_INDICES:-0 3 7}"
if [[ ${#source_indices[@]} -ne 3 ]]; then
  echo "SOURCE_INDICES must contain exactly three frame indices" >&2
  exit 1
fi

path_args=()
if [[ -n "${PATH_INDICES:-}" ]]; then
  read -r -a path_indices <<< "${PATH_INDICES}"
  path_args=(--path_indices "${path_indices[@]}")
fi

python3 inference/infer_video.py \
  "${scene_args[@]}" \
  --model_path="${MODEL_PATH}" \
  --vae_path="${VAE_PATH}" \
  --out_dir="${OUT_DIR:-inference_results/video}" \
  --model=1b \
  --use_prope=1 \
  --cos=0 \
  --N_views_src=3 \
  --src_indices "${source_indices[@]}" \
  --N_views_tgt="${CHUNK_SIZE:-7}" \
  --frames_per_segment="${FRAMES_PER_SEGMENT:-2}" \
  --fps="${FPS:-24}" \
  --cfg="${CFG:-1.0}" \
  --tau="${TAU:-0.5}" \
  --img_ext="${IMG_EXT:-webp}" \
  "${path_args[@]}"
