#!/bin/bash

cd ../pytorch
source config_G593-ZD1_01x08x32.sh
export CONT=./nvdlfwea+mlperfv41+sd.pytorch.sqsh
export DATADIR=/data/datasets/stable_diffusion/datasets
export CHECKPOINTS=/data/datasets/stable_diffusion/checkpoints
export LOGDIR=$(pwd)/../results
export NEMOLOGS=$(pwd)/../nemologs

sbatch -N $DGXNNODES -t $WALLTIME run.sub
