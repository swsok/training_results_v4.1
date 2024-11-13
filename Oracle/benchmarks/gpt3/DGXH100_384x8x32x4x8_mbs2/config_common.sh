## System run parms

export TRAIN_ONLY=0

# DL Params
export USE_DIST_OPTIMIZER=True

# This is to improve p2p overlap on H100
export NVTE_FWD_LAYERNORM_SM_MARGIN=8
export NVTE_BWD_LAYERNORM_SM_MARGIN=8

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_MIN_NCHANNELS=4

export CUDA_DEVICE_MAX_CONNECTIONS=1

: "${CHECKPOINT_NAME:=""}"
export LOAD_CHECKPOINT="/load_checkpoints/"$CHECKPOINT_NAME

export MICRO_BATCH_SIZE=2

: "${LOAD_MINIMAL_NUM_SAMPLES:=0}"

if [[ "${LOAD_MINIMAL_NUM_SAMPLES}" -eq 1 ]]; then
  export MAX_STEPS=500
  export OVERRIDE_ZERO_CONSUMED_SAMPLES=0
  export INIT_GLOBAL_STEP=0
fi

# This is needed to save memory. nvbug 4264087 tracks  fix.
export NCCL_NVLS_ENABLE=0

# TP overlap: use FP8/MC strided atomic RS and pipelined AG
export NVTE_UB_SPLIT_RS=0
export NVTE_UB_ATOMIC_GEMM_RS=1
export NVTE_RS_STRIDED_ATOMIC=1
#export NVTE_UB_FP8_RS=1
unset UB_SKIPMC
export MC_TP_OVERLAP_AG=True
export MC_TP_OVERLAP_RS=True

# FA: Disbale FAv2 from cuDNN and optimizations that consume memory (expected < 200MB) as they cause IMAs
#export NVTE_FUSED_ATTN=0 # Disable cuDNN fused attention
#export NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT=0

# Increase p2p chunksize to 2MB
export NCCL_P2P_NET_CHUNKSIZE=2097152

# Enable per-communicator nccl option tuning
export NCCL_CFG_PATH="/workspace/llm/conf/nccl/custom_communicator_cta.yaml"

# Disable gc when switching to/from validation steps
export NEMO_MANUAL_GC_IN_VALIDATION=0

# disable tensorboard logging
export EXTRA_ARGS="exp_manager.create_tensorboard_logger=False ${EXTRA_ARGS:-}"

# skip unnecessary broadcasting of training loss
export NEMO_LOG_TRAIN_LOSS=0

export BUCKET_CAP_MB=200

#SHARP
#export NCCL_SHARP_GROUP_SIZE_THRESH=2  #Avoid falling back to non-sharp

export FP8_PARAMS=True

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
export OMPI_MCA_btl_tcp_if_include="192.168.0.0/16"
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
      
MPIVARS_PATH=/nfs/scratch/mpi/hpcx-v2.20-gcc-mlnx_ofed-ubuntu22.04-cuda12-x86_64/hpcx-init-ompi.sh
LOCAL_MPI=/nfs/scratch/mpi/hpcx-v2.20-gcc-mlnx_ofed-ubuntu22.04-cuda12-x86_64
