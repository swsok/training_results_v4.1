# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## DL params

# tunable HPs
# 10 experiments show that 
# this config converges at around 2.50 epochs
export EPOCHS="3" 
export BATCH_SIZE="512" # local batch size
export VALIDATION_BATCH_SIZE="1024"
export LEARNING_RATE="0.005" # optimal LR at this config

# WG related
export WG_SHARDING_LOCATION="cuda"
export WG_SHARDING_PARTITION="global"
export WG_SHARDING_TYPE="distributed"
export SAMPLING_DEVICE="cuda"
export GRAPH_DEVICE="cuda"
export NUM_SAMPLING_THREADS="1"
export NUM_WORKERS="0"

# Knobs
export TRAIN_OVERLAP="1"
export EVAL_OVERLAP="1"
export HIGH_PRIORITY_EMBED_STREAM="1"
export PAD_NODE_COUNT_TO="3072"
export GC_THRESHOLD_MULTIPLIER="2"

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
export DGXNNODES=64
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=5  # measured: run_and_time.sh takes between 2 and 3 minutes
export WALLTIME=$((10 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
