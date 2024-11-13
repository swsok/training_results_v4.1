#!/bin/bash

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#source $(dirname ${BASH_SOURCE[0]})/config_XE8640_common.sh

# config_XE8640_common.sh
export WARMUP=True
export DGXNGPU=4
export WORLD_SIZE=${DGXNGPU}
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[1]}) | sed 's/^config_//' | sed 's/\.sh$//' )

#export NCCL_CFG_PATH="conf/nccl/custom_communicator_cta_4gpu.yaml"
#export NCCL_CFG_PATH="/data/custom_communicator_cta_4gpu.yaml"
export NCCL_MIN_P2P_NCHANNELS=32;
export NCCL_MIN_CTAS=32;
export NCCL_NCHANNELS_PER_NET_PEER=32;
#export NCCL_MIN_P2P_NCHANNELS=16;
#export NCCL_MIN_CTAS=16;
#export NCCL_NCHANNELS_PER_NET_PEER=16;
export TP_COMM_OVERLAP=False
export MC_TP_OVERLAP_AG=False
export MC_TP_OVERLAP_RS=False
export MC_TP_OVERLAP_RS_DGRAD=False
export CUBLAS_FORCE_XMMA_KERNEL_INIT=DEVICE
export NVTE_RS_STRIDED_ATOMIC=2
export LORA_A2A=1

export POSSIBLE_USER_WARNINGS=0
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1
export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 # Disable caching NCCL communication buffer
export NCCL_NVLS_ENABLE=0 # Disable NVL SHARP, which don't use
export CUDA_DEVICE_MAX_CONNECTIONS=1

export FP8=True
export FP8_AMAX_ALGO=max
export FP8_REDUCE_AMAX=False
export FP8_AMAX_HISTORY=32
export HYDRA_FULL_ERROR=1

export PP=1
export SP=1

# other
export MBS=1
export VAL_CHECK_INTERVAL=384

# hyperparameters
export MAX_STEPS=800
export LR=0.0004
export MINIBS=4
export TP=4
export CP=1
export TP_COMM_OVERLAP=0
export VBOOST_VALUE=1
export FP8_DPA=1
export NVTE_FP8_DPA_BWD=1
export SKIP_EVALS=3

# system parameters
export DGXNNODES=1
export WALLTIME_RUNANDTIME=80
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
