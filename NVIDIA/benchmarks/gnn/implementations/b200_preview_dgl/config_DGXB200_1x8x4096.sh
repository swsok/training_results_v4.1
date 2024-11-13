## DL params

# tunable HPs
# This configuration generally converges using 0.92~0.93 epochs
# if we hit 1 full epoch then we need to abort
export EPOCHS="1"
export BATCH_SIZE="4096" # local batch size
export VALIDATION_BATCH_SIZE="2048"
export LEARNING_RATE="0.002" 

# WG related
export WG_SHARDING_LOCATION="cuda"
export WG_SHARDING_PARTITION="node"
export WG_SHARDING_TYPE="continuous"
export WG_GATHER_SM="-1"
export SAMPLING_DEVICE="cuda"
export GRAPH_DEVICE="cuda"
export NUM_SAMPLING_THREADS="1"
export NUM_WORKERS="0"

# Knobs
export TRAIN_OVERLAP="1"
export EVAL_OVERLAP="1"
export HIGH_PRIORITY_EMBED_STREAM="0"
export USE_CONCAT_EMBEDDING="0"
export PAD_NODE_COUNT_TO="3072"
export WARMUP_MODEL="1"

export AMP="1"
export DIST_ADAM="1"

# CUDA Graph related
export USE_CUDA_GRAPH="1"
export CUDA_GRAPH_ESTIMATION_BATCHES="20"
export CUDA_GRAPH_PADDING_SIGMA="5"

# Configs that should not change
export FAN_OUT="5,10,15"
export HIDDEN_DIM="512"
export NUM_HEADS="4"
export EVAL_FREQUENCY="0.05"

# debugging
export TIMETAG="1"
export DEBUG="1"

## System run params
export DGXNNODES=1
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=30  # measured: run_and_time.sh takes up to 20 minutes
export WALLTIME=$((10 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
