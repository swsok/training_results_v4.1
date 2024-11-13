export DATADIR="/mnt/training_ds/stable_diff_ds/datasets"
export CHECKPOINTS="/mnt/training_ds/stable_diff_ds/checkpoints"
export NEMOLOGS="/mnt/training_ds/stable_diff_ds/nemologs"  # needed for intermediate results between training and evaluation
export LOGDIR=`pwd`/results  # needed to save mlperf results (output logs)
export CONT=nvcr.io/nvdlfwea/mlperfv41/sd:20240923.pytorch
export DGXNNODES=1
export WALLTIME=12000
export NEXP=10
source config_XE8640.sh  # or any other config
sbatch -N $DGXNNODES -t $WALLTIME run.sub
