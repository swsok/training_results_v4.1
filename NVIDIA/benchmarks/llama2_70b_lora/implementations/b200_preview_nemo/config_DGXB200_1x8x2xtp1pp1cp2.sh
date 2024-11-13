#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters
export MAX_STEPS=800
export LR=0.0004
export MINIBS=2
export TP=1
export PP=1
export SP=0
export CP=2
export SKIP_EVALS=3

export HYDRA_FULL_ERROR=1
export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0
export VBOOST_VALUE=0

export FP8=True
export FP8_AMAX_ALGO=max
export FP8_REDUCE_AMAX=True
export FP8_AMAX_HISTORY=4
export FP8_ACTIVATION=False

export TP_COMM_OVERLAP=True
export UB_SKIPMC=1
export NCCL_MIN_CTAS=32

export UCX_TLS=self,tcp

export NVTE_UB_ATOMIC_GEMM_RS_PROJ=0
export NVTE_UB_ATOMIC_GEMM_RS_FC2=0

# to work-around cudnn bug on blackwell
export FP8_DPA=1
export NVTE_FP8_DPA_BWD=1
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WARMUP_TRAIN_STEPS=5
export WARMUP=True
export WARMUP_VALIDATION_STEPS=5
export MBS=1
export MCORE_OPTIM_OVERLAP_PARAM_SYNC=True  # True or False
export NEMO_LORA_MEMORY_SAVING_DROPOUT=0    # 1 or 0
export LAYER_CUDA_GRAPH=1
export CG_WEIGHT_CACHING=False

# system parameters
# export LOAD_CKPT=True
export DGXNNODES=1
export DGXNGPU=8
export WALLTIME_RUNANDTIME=50
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
