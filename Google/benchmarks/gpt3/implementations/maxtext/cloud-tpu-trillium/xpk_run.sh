#!/bin/bash

# Prerequiste: pip install xpk
# Example:
# WORKLOAD_NAME=${USER}-trillium-512 SCRIPT=trillium-512.sh NUM_SLICES=2 bash xpk_run.sh

SCRIPTS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd )"

set -euox pipefail
PROJECT=${PROJECT:-some-cloud-project-id}
ZONE=${ZONE:-europe-west4-a}
CLUSTER_NAME=${CLUSTER_NAME:-mlperf-v6e-512-europe-west4-a}
DEVICE_TYPE=${DEVICE_TYPE:-v6e-256}
NUM_SLICES=${NUM_SLICES:-2}

# in maxtext repo
# bash docker_build_dependency_image.sh MODE=nightly
BASE_IMAGE=${BASEIMAGE:-gcr.io/cloud-tpu-v2-images/maxtext_jax_nightly:2024-10-10}

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
WORKLOAD_NAME=${WORKLOAD_NAME:-${USER}-run}

BASE_OUTPUT_DIRECTORY=${BASE_OUTPUT_DIRECTORY:-"gs://some-bucket"}

# One of trillium-512.sh, trillium-1024.sh, trillium-2048.sh, trillium-3072.sh
SCRIPT=${SCRIPT:-trillium-512.sh}

# xpk will pack current directory into a docker image, go to the script directory first
pushd "${SCRIPTS_DIR}"
python ~/xpk/xpk.py workload create --cluster="${CLUSTER_NAME}" --workload="${WORKLOAD_NAME}-${TIMESTAMP}" --device-type="${DEVICE_TYPE}"\
  --num-slices="${NUM_SLICES}" --zone="${ZONE}" --project="${PROJECT}" --base-docker-image "${BASE_IMAGE}"\
  --command "BASE_OUTPUT_DIRECTORY=${BASE_OUTPUT_DIRECTORY} WORKLOAD_NAME=${WORKLOAD_NAME} TIMESTAMP=${TIMESTAMP} USER=${USER} bash ${SCRIPT}"
popd
