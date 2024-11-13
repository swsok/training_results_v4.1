# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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

export DGXNNODES=1
export DGXNGPU=16
export BATCHSIZE=64
export CONFIG_MAX_STEPS=3500
export INFER_START_STEP=2000

export BASE_LR=${BASE_LR:-"0.00000011"}
export WARMUP_STEPS=500

export FLASH_ATTENTION=${FLASH_ATTENTION:-False}
export USE_TE_DPA=${USE_TE_DPA:-True}
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1

export WALLTIME_RUNANDTIME=180

# Set clocks and walltime for maxQ and minEDP runs
if [[ "${SET_MAXQ_CLK:-0}" == "1" ]]; then
  export MAXQ_CLK=1050
  WALLTIME_RUNANDTIME=$(expr ${WALLTIME_RUNANDTIME} + ${WALLTIME_RUNANDTIME} / 2) # 50% longer walltime
elif [[ "${SET_MINEDP_CLK:-0}" == "1" ]]; then
  export MINEDP_CLK=1290
  WALLTIME_RUNANDTIME=$(expr ${WALLTIME_RUNANDTIME} + ${WALLTIME_RUNANDTIME} / 3) # 33% longer walltime
fi

export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

# Load default settings
source $(dirname ${BASH_SOURCE[0]})/config_common.sh
