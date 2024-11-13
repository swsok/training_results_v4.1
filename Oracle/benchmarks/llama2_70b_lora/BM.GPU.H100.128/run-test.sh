source /mnt/orangefs/mlperf/llama/code/configs/config_DGXH100_16x8x1xtp4pp1cp2.sh
export LOGDIR=/mnt/orangefs/mlperf/llama/logs
export DATADIR=/mnt/orangefs/mlperf/llama/data_model/gov_report
export MODEL=/mnt/orangefs/mlperf/llama/data_model/model
export WALLTIME=300

#export EXCLUDELIST="GPU-215,GPU-639"

#CONT=/mnt/localdisk/sd/mlperf/build/loraubuntu.sqsh \
CONT=/mnt/orangefs/mlperf/llama/cont/loraubuntu.sqsh \
NCCL_TEST=0 \
MLPERF_SYSTEM_NAME="BM.GPU.H100.8" \
MLPERF_SUBMITTER="Oracle" \
MLPERF_STATUS="cloud" \
MLPERF_DIVISION="closed" \
MLPERF_CLUSTER_NAME="BM.GPU.H100.8 Cluster" \
sbatch -W -p compute -N $DGXNNODES -t $WALLTIME run.sub

#sbatch -p compute -N $DGXNNODES -t $WALLTIME --exclude $EXCLUDELIST run.sub
