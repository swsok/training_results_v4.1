#!/bin/bash
export WARMUP=True
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[1]}) | sed 's/^config_//' | sed 's/\.sh$//' )

# export NCCL_CFG_PATH="conf/nccl/custom_communicator_cta.yaml"
export NCCL_MIN_P2P_NCHANNELS=32
export NCCL_MIN_CTAS=32
export NCCL_NCHANNELS_PER_NET_PEER=32
export TP_COMM_OVERLAP=True
export MC_TP_OVERLAP_AG=True
export MC_TP_OVERLAP_RS=True
export MC_TP_OVERLAP_RS_DGRAD=False
export MC_TP_DISABLE_QKV=False
export CUBLAS_FORCE_XMMA_KERNEL_INIT=DEVICE
export NVTE_RS_STRIDED_ATOMIC=2
export LORA_A2A=1

export FP8_DPA=1
export NVTE_FP8_DPA_BWD=1

export POSSIBLE_USER_WARNINGS=0
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1
export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 # Disable caching NCCL communication buffer
export NCCL_NVLS_ENABLE=0 # Disable NVL SHARP, which don't use
# other
export MBS=1
export SKIP_EVALS=4
export VAL_CHECK_INTERVAL=384


export SUBMISSION_ORG="LAMBDA"
export MLPERF_SUBMITTER=$SUBMISSION_ORG
export MLPERF_CLUSTER_NAME="lambda_1cc"
export MLPERF_SYSTEM_NAME=$MLPERF_CLUSTER_NAME
export MLPERF_STATUS=onprem
export MLPERF_DIVISION=closed

# Lambda
export DGXSOCKETCORES=48
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export CONT="$(hostname):5000#local/mlperf-nvidia-llama2_70b_lora:latest"
export DATADIR="/home/ubuntu/ml-1cc/data/mlperf/llama2_70b_lora/data"
export MODEL="/home/ubuntu/ml-1cc/data/mlperf/llama2_70b_lora/ckpt"
export NEXP=10
