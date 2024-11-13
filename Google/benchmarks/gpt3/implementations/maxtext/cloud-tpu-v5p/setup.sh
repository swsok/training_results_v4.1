#!/bin/bash
ZONE=us-east5-c
TPU_TYPE=v5p-4096  # can be any one of v5p-4096, v5p-8192, v5p-12288
CLUSTER_NAME="mlperf-${TPU_TYPE}-${ZONE}"
PROJECT=some-cloud-tpu-project-id
NUM_SLICES=1

# cluster setup
python3 xpk.py cluster create --cluster "${CLUSTER_NAME}" \
  --num-slices="${NUM_SLICES}" --tpu-type="${TPU_TYPE}" --zone="${ZONE}" \
  --project="${PROJECT}" --on-demand