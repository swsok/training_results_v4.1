## System config params
export DGXNGPU=8
export DGXSOCKETCORES=96
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}
export CONT="$(hostname):5000#local/mlperf-nvidia-bert:latest"
export NEXP=10