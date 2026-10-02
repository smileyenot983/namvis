#!/usr/bin/env bash

set -x
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# set dist args
# SINGLE=1
# nproc_per_node=${ARNOLD_WORKER_GPU}

if [[ "${SINGLE:-1}" != "0" ]]; then
  echo "[single node alone] SINGLE=${SINGLE:-1}"
  nnodes=1
  node_rank=0
  nproc_per_node="${NPROC_PER_NODE:-1}"
  master_addr=127.0.0.1
  master_port=6661
else
  MASTER_NODE_ID=0
  nnodes=${ARNOLD_WORKER_NUM}
  node_rank=${ARNOLD_ID}
  master_addr="METIS_WORKER_${MASTER_NODE_ID}_HOST"
  master_addr=${!master_addr}
  master_port="METIS_WORKER_${MASTER_NODE_ID}_PORT"
  master_port=${!master_port}
  ports=(`echo $master_port | tr ',' ' '`)
  master_port=${ports[0]}
fi

echo "[nproc_per_node: ${nproc_per_node}]"
echo "[nnodes: ${nnodes}]"
echo "[node_rank: ${node_rank}]"
echo "[master_addr: ${master_addr}]"
echo "[master_port: ${master_port}]"

# set up envs
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export TRACKIO_DIR="${REPO_ROOT}/trackio_logs"

BED=checkpoints
LOCAL_OUT=local_output
mkdir -p $BED
mkdir -p $LOCAL_OUT

export COMPILE_GAN=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export RUN_NAME="${RUN_NAME:-co3d_prope_1b_image}"
export project_name=infinity3d

wandb offline
exp_name=debug
bed_path=checkpoints/${RUN_NAME}/

export data_path="/workspace/data/objaverse_sketchfab_web_pmap1,/workspace/data/objaverse_sketchfab_web_pmap2,/workspace/data/objaverse_github_web_pmap"
export eval_path="${EVAL_DATA_PATH:-/workspace/namvis/data_eval/rendered_objaverse8_wdepth_pitch30/}"

export eval_json="${EVAL_JSON:-data_eval/eval_objaverse_small.json}"


video_data_path=''
local_out_path=$LOCAL_OUT/${RUN_NAME}

out_dir=outputs_${RUN_NAME}


mkdir -p "${out_dir}" "${bed_path}" "${local_out_path}"

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# export NCCL_IB_DISABLE=1           # disable InfiniBand (force TCP)
unset NCCL_SOCKET_IFNAME           # let NCCL auto-detect network interfaces


echo "Shell: CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'"
echo "Shell: CUDA_DEVICE_ORDER='${CUDA_DEVICE_ORDER:-<unset>}'"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv

torchrun \
--nproc_per_node=${nproc_per_node} \
--nnodes=${nnodes} \
--node_rank=${node_rank} \
--master_addr=${master_addr} \
--master_port=${master_port} \
train.py \
--ep=10000 \
--opt=adamw \
--cum=3 \
--sche=lin0 \
--fp16=2 \
--ada=0.9_0.97 \
--tini=-1 \
--tclip=5 \
--flash=0 \
--alng=5e-06 \
--saln=1 \
--cos=0 \
--enable_checkpointing=full-block \
--local_out_path ${local_out_path} \
--task_type='t2i' \
--bed=${bed_path} \
--data_path=${data_path} \
--eval_path="${eval_path}" \
--eval_json="${eval_json}" \
--eval_backend=rendered \
--eval_max_scenes="${EVAL_MAX_SCENES:-100}" \
--eval_cfg=1.0 \
--eval_tau=0.5 \
--eval_seed=0 \
--eval_freq="${EVAL_FREQ:-1}" \
--train_scenes=218186 \
--video_data_path=${video_data_path} \
--exp_name=${exp_name} \
--tblr=6e-3 \
--pn 0.06M \
--model=1bc8 \
--lbs=2 \
--workers=1 \
--short_cap_prob 0.5 \
--online_t5=0 \
--use_streaming_dataset 1 \
--iterable_data_buffersize 30000 \
--Ct5=2048 \
--t5_path="${T5_PATH:-weights/flan_t5}" \
--vae_type 32 \
--vae_ckpt="${VAE_CKPT:-weights/infinity_vae_d32reg.pth}" \
--rush_resume="${RUSH_RESUME:-weights/namvis_1b.pth}" \
"${resume_args[@]}" \
--wp 0.00000001 \
--wpe=1 \
--dynamic_resolution_across_gpus 1 \
--enable_dynamic_length_prompt 1 \
--reweight_loss_by_scale 1 \
--add_lvl_embeding_only_first_block 1 \
--rope2d_each_sa_layer 1 \
--rope2d_normalized_by_hw 2 \
--use_fsdp_model_ema 0 \
--always_training_scales 100 \
--use_bit_label 1 \
--zero=2 \
--save_model_iters_freq 5000 \
--log_freq=50 \
--checkpoint_type='torch' \
--prefetch_factor=16 \
--noise_apply_strength 0.3 \
--noise_apply_layers 13 \
--apply_spatial_patchify 0 \
--use_flex_attn=False \
--pad=0 \
--multiview=True \
--src2sos=True \
--N_views_src=2 \
--N_views_tgt=3 \
--out_dir=${out_dir} \
--run_name=${RUN_NAME} \
--use_prope=True \
--sos_source="image" \
--logger_name="trackio" \
