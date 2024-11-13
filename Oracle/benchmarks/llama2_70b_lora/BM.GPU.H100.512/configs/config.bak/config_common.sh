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

export WARMUP=True
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[1]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export NCCL_MIN_P2P_NCHANNELS=32;
export NCCL_MIN_CTAS=32;
export NCCL_NCHANNELS_PER_NET_PEER=32;
export TP_COMM_OVERLAP=True
export MC_TP_OVERLAP_AG=True
export MC_TP_OVERLAP_RS=True
export MC_TP_OVERLAP_RS_DGRAD=True
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

export NCCL_WORK_FIFO_DEPTH=1048576

# Enable SHARP for large scale only
#if [[ "${DGXNNODES}" -gt 128 ]]; then
#  export SHARP=True
#fi

# Use legacy NeMo dataset path
export LEGACY_DATASET=True

#---------------------- content from Sesh -------------------
set -eux
export PMI_DEBUG=1
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib
export OMPI_MCA_btl_tcp_if_include="10.224.0.0/12"
export PMIX_MCA_gds="^ds12" \
      NCCL_SOCKET_NTHREADS=16 \
      NCCL_DEBUG=WARN \
      NCCL_CUMEM_ENABLE=0 \
      NCCL_IB_SPLIT_DATA_ON_QPS=0 \
      NCCL_IB_QPS_PER_CONNECTION=1 \
      NCCL_IB_GID_INDEX=3 \
      NCCL_IB_TC=41 \
      NCCL_IB_SL=0 \
      NCCL_IB_TIMEOUT=22 \
      NCCL_NET_PLUGIN=none \
      NCCL_SOCKET_IFNAME=eth0 \
      NCCL_IGNORE_CPU_AFFINITY=1 \
      RX_QUEUE_LEN=8192 \
      IB_RX_QUEUE_LEN=8192 \
      UCX_NET_DEVICES=eth0 \
      UCX_TLS=tcp \
      HCOLL_ENABLE_MCAST_ALL=0 \
      coll_hcoll_enable=0 \
      NCCL_IB_HCA='=mlx5_0,mlx5_1,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_12,mlx5_13,mlx5_14,mlx5_15,mlx5_16,mlx5_17'
      
#MPIVARS_PATH=/nfs/scratch/mpi/hpcx-v2.20-gcc-mlnx_ofed-ubuntu22.04-cuda12-x86_64/hpcx-init-ompi.sh
#LOCAL_MPI=/nfs/scratch/mpi/hpcx-v2.20-gcc-mlnx_ofed-ubuntu22.04-cuda12-x86_64
#MPIVARS_PATH=/opt/hpcx-v2.16-gcc-mlnx_ofed-ubuntu22.04-cuda12-gdrcopy2-nccl2.18-x86_64/hpcx-init-ompi.sh
#LOCAL_MPI=/opt/hpcx-v2.16-gcc-mlnx_ofed-ubuntu22.04-cuda12-gdrcopy2-nccl2.18-x86_64
MPIVARS_PATH=/mnt/orangefs/mpi/hpcx-v2.20-gcc-mlnx_ofed-ubuntu22.04-cuda12-x86_64/hpcx-init-ompi.sh
LOCAL_MPI=/mnt/orangefs/mpi/hpcx-v2.20-gcc-mlnx_ofed-ubuntu22.04-cuda12-x86_64


