#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common_1cc.sh

# hyperparameters
export MAX_STEPS=1024
export LR=0.00036
export MINIBS=1

export TP=4
export PP=1
export CP=1
export SP=1
export UCX_TLS=self,shm,rc,ud,tcp

export FP8=True
export FP8_AMAX_ALGO=max
export FP8_REDUCE_AMAX=True
export FP8_AMAX_HISTORY=32

export SKIP_EVALS=3
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# system parameters
export DGXNNODES=4
export WALLTIME_MINUTES=45
export WALLTIME=$(( (${NEXP:-1} * WALLTIME_MINUTES) ))
export MLPERF_NUM_NODES=$DGXNNODES
export SBATCH_NETWORK=sharp
export SHARP=True
export NCCL_TEST=0

timestamp=$(date +'%y-%m-%d_%H-%M-%S')
export LOGDIR=./results/1cc_4x8x1xtp4pp1cp1_${timestamp}
mkdir -p ${LOGDIR}