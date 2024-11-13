#! /bin/bash

export DATADIR=/mnt/data4/work/stable_diffusion/datasets
export CHECKPOINTS=/mnt/data4/work/stable_diffusion/checkpoints
export CACHEDIR=/mnt/data4/work/transformers_cache/
export NEMOLOGS=$(realpath ../../logs/sd_nemo) # needed for intermediate results between training and evaluation
export LOGDIR=$(realpath ../../logs/sd) # needed to save mlperf results (output logs)

export CONT=$(realpath ../../sqsh_images/nvcr.io+nvdlfwea+mlperfv41+sd+20240923.pytorch.sqsh)
source config_PGCDI_01x16x64.sh
export PATH=/opt/slurm-23.11.10/bin:$PATH
export CKPT_PATH=/nemologs/stable-diffusion2-train-241004213222132455363-01/checkpoints/
export NEXP=1
export INFER_START_STEP=0
sbatch -N $DGXNNODES -t $WALLTIME run_eval.sub  # you may be required to set --account and --partition here
