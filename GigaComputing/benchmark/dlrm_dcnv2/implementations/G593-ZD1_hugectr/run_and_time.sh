#!/bin/bash

cd ../hugectr
export CONT=./nvdlfwea+mlperfv41+dlrm.hugectr.sqsh
source config_G593-ZD1_1x8x6912.sh  # use appropriate config
export DATADIR=/path/to/train/criteo_multihot_raw
export DATADIR_VAL=/path/to/val/criteo_multihot_raw

sbatch -N $DGXNNODES run.sub
