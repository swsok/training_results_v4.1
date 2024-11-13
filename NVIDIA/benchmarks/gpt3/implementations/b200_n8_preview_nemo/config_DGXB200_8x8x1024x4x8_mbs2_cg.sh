# 1n test config

## DL params
export MINIBS=1024
export TENSOR_MODEL_PARALLEL=4   #  training.model.tensor_model_parallel_size
export PIPELINE_MODEL_PARALLEL=8 #  training.model.pipeline_model_parallel_size

#=======================================================================
## System run parms
export DGXNNODES=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=470
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_blackwell.sh

export INTERLEAVED_PIPELINE=6   # interleaved pipeline size 0 to disable since PP=1
export MICRO_BATCH_SIZE=2

source $(dirname ${BASH_SOURCE[0]})/config_fp8.sh

export TP_COMM_OVERLAP=True

export NVTE_UB_FP8_RS=1

export LAYER_CUDA_GRAPH=1

export NVTE_RS_STRIDED_ATOMIC=0
export TE_UB_ATOMIC_GEMM_RS_PROJ=0
export TE_UB_ATOMIC_GEMM_RS_FC2=0

export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1
export HANG_MONITOR_TIMEOUT=7
