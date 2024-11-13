#!/bin/bash

cd ../pytorch
export DATADIR="<path/to/dir/containing/openimages/dir>"
export BACKBONE_DIR="<path/to/pretrained/ckpt>"
export LOGDIR="<path/to/output/dir>"
export CONT=./nvcr.io/nvdlfwea/mlperfv41/ssd.pytorch
source config_G593-ZD1_001x08x032.sh  # or any other config
sbatch -N $DGXNNODES -t $WALLTIME run.sub
