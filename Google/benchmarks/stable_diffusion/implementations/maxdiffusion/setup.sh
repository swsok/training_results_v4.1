#!/bin/bash
ZONE=${ZONE}
CLUSTER_NAME=mlperf-${TPU_TYPE}
NETWORK_NAME="${CLUSTER_NAME}-mtu9k"
NETWORK_FW_NAME="${NETWORK_NAME}-fw"
PROJECT=${PROJECT}
NUM_SLICES=1

# network setup
gcloud compute networks create "${NETWORK_NAME}" --mtu=8896 --project="${PROJECT}" --subnet-mode=auto --bgp-routing-mode=regional
gcloud compute firewall-rules create "${NETWORK_FW_NAME}" --network "${NETWORK_NAME}" --allow tcp,icmp,udp --project="${PROJECT}"

TPU_TYPE=v5p-16
# cluster setup
export CLUSTER_ARGUMENTS="--network=${NETWORK_NAME} --subnetwork=${NETWORK_NAME}"
python3 xpk.py cluster create --cluster "${CLUSTER_NAME}" \
  --num-slices="${NUM_SLICES}" --tpu-type="${TPU_TYPE}" --zone="${ZONE}" \
  --project="${PROJECT}" --on-demand \
  --custom-cluster-arguments="${CLUSTER_ARGUMENTS}"

TPU_TYPE=v5p-128
# cluster setup
export CLUSTER_ARGUMENTS="--network=${NETWORK_NAME} --subnetwork=${NETWORK_NAME}"
python3 xpk.py cluster create --cluster "${CLUSTER_NAME}" \
  --num-slices="${NUM_SLICES}" --tpu-type="${TPU_TYPE}" --zone="${ZONE}" \
  --project="${PROJECT}" --on-demand \
  --custom-cluster-arguments="${CLUSTER_ARGUMENTS}"

TPU_TYPE=v5p-1024
# cluster setup
export CLUSTER_ARGUMENTS="--network=${NETWORK_NAME} --subnetwork=${NETWORK_NAME}"
python3 xpk.py cluster create --cluster "${CLUSTER_NAME}" \
  --num-slices="${NUM_SLICES}" --tpu-type="${TPU_TYPE}" --zone="${ZONE}" \
  --project="${PROJECT}" --on-demand \
  --custom-cluster-arguments="${CLUSTER_ARGUMENTS}"

TPU_TYPE=v5p-2048
# cluster setup
export CLUSTER_ARGUMENTS="--network=${NETWORK_NAME} --subnetwork=${NETWORK_NAME}"
python3 xpk.py cluster create --cluster "${CLUSTER_NAME}" \
  --num-slices="${NUM_SLICES}" --tpu-type="${TPU_TYPE}" --zone="${ZONE}" \
  --project="${PROJECT}" --on-demand \
  --custom-cluster-arguments="${CLUSTER_ARGUMENTS}"