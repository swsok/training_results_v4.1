#! /bin/bash

export EVALDIR=/mnt/data4/work/bert_data/hdf5/eval_varlength
export DATADIR_PHASE2="/mnt/data4/work/bert_data/hdf5/training-4320/hdf5_4320_shards_varlength_shuffled"
export DATADIR_PHASE2_PACKED="/mnt/data4/work/bert_data/packed_data"
export CHECKPOINTDIR_PHASE1="/mnt/data4/work/bert_data/phase1"
export LOGDIR=$(realpath ../../logs/bert)
export CONT=$(realpath ../../sqsh_images/nvcr.io+nvdlfwea+mlperfv41+bert+20240923.pytorch.sqsh)
#export CONT=$(realpath ../../sqsh_images/mlperf-nvidia+language_model-2409-pyt.sqsh)
export CONTAINER_PRELOAD_LUSTRE=0
#export PATH=/opt/slurm-23.11.10/bin:$PATH
export PATH=/opt/slurm-23.11.10-pmix3/bin:$PATH
export NEXP=10
source config_PGCDI_1x16x48x1_pack.sh
sbatch -N ${DGXNNODES} --time=${WALLTIME} run.sub  # you may be required to set --account and --partition here
