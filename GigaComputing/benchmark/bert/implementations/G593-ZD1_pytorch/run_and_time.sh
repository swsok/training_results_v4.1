#!/bin/bash

cd ../pytorch
export EVALDIR="/path/to/your/data/hdf5/eval_varlength"
export DATADIR_PHASE2="/path/to/your/data/hdf5/training-4320/hdf5_4320_shards_varlength_shuffled"
export DATADIR_PHASE2_PACKED="/path/to/your/data/packed_data"
export CHECKPOINTDIR_PHASE1="/path/to/your/data/phase1"
export LOGDIR=</path/to/output/dir> # set the place where the output logs will be saved
export CONT=./nvcr.io/nvdlfwea/mlperfv41/bert.pytorch
source config_G593-ZD1_1x8x48x1_pack.sh  # select config
export CONTAINER_PRELOAD_LUSTRE=0

sbatch -N${DGXNNODES} --ntasks-per-node=${DGXNGPU} --time=${WALLTIME} run.sub
