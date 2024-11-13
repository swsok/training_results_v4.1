#!/bin/bash
set -x 

#export DATADIR=</path/to/dataset>/gov_report # set correct </path/to/dataset>
#export MODEL=</path/to/dataset>/model # set correct </path/to/dataset>
#export LOGDIR=</path/to/output/dir> # set the place where the output logs will be saved
#export CONT=nvcr.io/nvdlfwea/mlperfv41/lora:20240918.pytorch
#source configs/config_DGXH100_1x8x8x4x2_fp8.sh  # use appropriate config
#sbatch -N $DGXNNODES -t $WALLTIME run.sub



#export DATADIR=/mnt/dlrmv2_ds/training_ds/Llama2/dataset/scrolls_gov_report_8k # set correct </path/to/dataset>
#export MODEL=/mnt/dlrmv2_ds/training_ds/Llama2/models/Llama2-70b-fused-qkv-mlperf # set correct </path/to/dataset>
#export LOGDIR=/mnt/dlrmv2_ds/train_workdir/091924/llama2/logs # set the place where the output logs will be saved
#export CONT=nvcr.io/nvdlfwea/mlperfv41/lora:20240918.pytorch
#source configs/config_XE8640_1x4x4xtp4pp1cp1.sh  # use appropriate config
#sbatch -N $DGXNNODES -t $WALLTIME run.sub

export PREFIX=/mnt/dlrmv2_ds/training_ds/Llama2
export DGXNNODES=1
export DATADIR=$PREFIX/gov_report # set correct </path/to/dataset>
export MODEL=$PREFIX/model # set correct </path/to/dataset>
export LOGDIR=/mnt/dlrmv2_ds/train_workdir/091924/llama2/logs # set the place where the output logs will be saved
export CONT=nvcr.io/nvdlfwea/mlperfv41/lora:20240918.pytorch
export NEXP=10
#source configs/config_XE8640_1x4x4xtp4pp1cp1.sh  # use appropriate config
source configs/config_XE8640_1x4x4xtp4pp1cp1_2.sh
#sbatch -N $DGXNNODES -t $WALLTIME run_orig.sub
sbatch -N $DGXNNODES -t $WALLTIME run3.sub

