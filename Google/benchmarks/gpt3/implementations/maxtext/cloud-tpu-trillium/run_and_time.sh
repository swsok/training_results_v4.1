#!/bin/bash
TPU_TYPE=${TPU_TYPE:-v6e-256}
ZONE=europe-west4-a
export CLUSTER_NAME="mlperf-${TPU_TYPE}-${ZONE}"
export PROJECT=some-cloud-tpu-project-id

WORKLOAD_NAME="${USER}-${TPU_TYPE}" SCRIPT="{TPU_TYPE}.sh" bash xpk_run.sh