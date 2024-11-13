#!/bin/bash
TPU_TYPE=${TPU_TYPE:-v5p-4096}
ZONE=us-east5-c
export CLUSTER_NAME="mlperf-${TPU_TYPE}-${ZONE}"
export PROJECT=some-cloud-tpu-project-id

WORKLOAD_NAME="${USER}-${TPU_TYPE}" DEVICE_TYPE="{TPU_TYPE}" SCRIPT="{TPU_TYPE}.sh" bash xpk_run.sh