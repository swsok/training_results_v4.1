#! /bin/bash

export DATADIR=/mnt/data4/work/stable_diffusion/datasets
export CHECKPOINTS=/mnt/data4/work/stable_diffusion/checkpoints
export CACHEDIR=/mnt/data4/work/transformers_cache/
export NEMOLOGS=$(realpath ../../logs/sd_nemo) # needed for intermediate results between training and evaluation
export LOGDIR=$(realpath ../../logs/sd) # needed to save mlperf results (output logs)

export CONT=$(realpath ../../sqsh_images/nvcr.io+nvdlfwea+mlperfv41+sd+20240923.pytorch.sqsh)
export NEXP=10
source config_PGCDI_01x16x64.sh
export PATH=/opt/slurm-23.11.10/bin:$PATH
sbatch -N $DGXNNODES -t $WALLTIME run.sub  # you may be required to set --account and --partition here
