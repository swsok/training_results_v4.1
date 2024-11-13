#! /bin/bash

export DATADIR=/mnt/data4/work/ssd-openimages
export BACKBONE_DIR=/mnt/data4/work/ssd-backbone
export LOGDIR=$(realpath ../../logs/ssd)
export CONT=$(realpath ../../sqsh_images/nvcr.io+nvdlfwea+mlperfv41+ssd+20240923.pytorch.sqsh)
export NEXP=5
source config_PGCDI_001x16x016.sh  # select config and source it
export PATH=/opt/slurm-23.11.10/bin:$PATH
export VERIFY_MOUNT=0
sbatch -N $DGXNNODES -t $WALLTIME run.sub  # you may be required to set --account and --partition here
