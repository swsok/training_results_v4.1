#!/bin/bash

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

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters
export LR=0.0005
export MAX_STEPS=1024
export MINIBS=1 
export TP=4
export CP=2
export TP_COMM_OVERLAP=1
export FP8_DPA=0
export NVTE_FP8_DPA_BWD=0
export SKIP_EVALS=18
export LAYER_CUDA_GRAPH=1
export MC_TP_OVERLAP_RS_DGRAD=False

# system parameters
export DGXNNODES=1
export WALLTIME_RUNANDTIME=15
export SBATCH_NETWORK=sharp
export SHARP=False
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
