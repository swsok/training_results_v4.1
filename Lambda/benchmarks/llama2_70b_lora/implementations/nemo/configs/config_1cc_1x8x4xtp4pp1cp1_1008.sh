#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common_1cc_1008.sh

# hyperparameters
export MAX_STEPS=2000
export LR=0.0005
export MINIBS=4

export TP=4
export PP=1
export CP=1
export SP=1
export TP_COMM_OVERLAP=False
export VBOOST_VALUE=1

export FP8=True
export FP8_AMAX_ALGO=max
export FP8_REDUCE_AMAX=False
export FP8_AMAX_HISTORY=32

export SKIP_EVALS=3
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# system parameters
export DGXNNODES=1
export WALLTIME_MINUTES=45
export WALLTIME=$(( (${NEXP:-1} * WALLTIME_MINUTES) ))
export MLPERF_NUM_NODES=$DGXNNODES

timestamp=$(date +'%y-%m-%d_%H-%M-%S')
export LOGDIR=./results/1cc_1x8x4xtp4pp1cp1_1008_${timestamp}
mkdir -p ${LOGDIR}