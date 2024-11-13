#!/bin/bash

cd ../pytorch
export DATADIR=</path/to/dataset>/gov_report # set correct </path/to/dataset>
export MODEL=</path/to/dataset>/model # set correct </path/to/dataset>
export LOGDIR=</path/to/output/dir> # set the place where the output logs will be saved
export CONT=./nvcr.io/nvdlfwea/mlperfv41/lora.pytorch
source configs/config_G593-ZD1_1x8x2xtp2pp1cp1.sh  # use appropriate config
sbatch -N $DGXNNODES -t $WALLTIME run.sub
