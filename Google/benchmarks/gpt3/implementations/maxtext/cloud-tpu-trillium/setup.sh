#!/bin/bash
ZONE=europe-west4-a
TPU_TYPE=v6e-256
CLUSTER_NAME="mlperf-${TPU_TYPE}-${ZONE}"
PROJECT=some-cloud-tpu-project-id
NUM_SLICES=2

NETWORK_NAME_1=${CLUSTER_NAME}-mtu9k-1-${ZONE}
NETWORK_FW_NAME_1=${NETWORK_NAME_1}-fw-1-${ZONE}

# Use a custom network for better performance as well as avoid the default network to be overloaded.
gcloud compute networks create ${NETWORK_NAME_1} --mtu=8896 --project=${PROJECT} --subnet-mode=auto --bgp-routing-mode=regional
gcloud compute firewall-rules create ${NETWORK_FW_NAME_1} --network ${NETWORK_NAME_1} --allow tcp,icmp,udp --project=${PROJECT}

# Secondary subnet for multinic experience. Need custom ip routing to be different from first network’s subnet.
export NETWORK_NAME_2=${CLUSTER_NAME}-privatenetwork-2-${ZONE}
export SUBNET_NAME_2=${CLUSTER_NAME}-privatesubnet-2-${ZONE}
export FIREWALL_RULE_NAME=${CLUSTER_NAME}-privatefirewall-2-${ZONE}
export ROUTER_NAME=${CLUSTER_NAME}-network-2-${ZONE}
export NAT_CONFIG=${CLUSTER_NAME}-natconfig-2-${ZONE}
export REGION=us-central2
gcloud compute networks create "${NETWORK_NAME_2}" --mtu=8896 --bgp-routing-mode=regional --subnet-mode=custom --project=$PROJECT
gcloud compute networks subnets create "${SUBNET_NAME_2}" --network="${NETWORK_NAME_2}" --range=10.10.0.0/18 --region="${REGION}" --project=$PROJECT
gcloud compute firewall-rules create "${FIREWALL_RULE_NAME}" --network "${NETWORK_NAME_2}" --allow tcp,icmp,udp --project="${PROJECT}"
gcloud compute routers create "${ROUTER_NAME}" \
  --project="${PROJECT}" \
  --network="${NETWORK_NAME_2}" \
  --region="${REGION}"
gcloud compute routers nats create "${NAT_CONFIG}" \
  --router="${ROUTER_NAME}" \
  --region="${REGION}" \
  --auto-allocate-nat-external-ips \
  --nat-all-subnet-ip-ranges \
  --project="${PROJECT}" \
  --enable-logging

export CLUSTER_ARGUMENTS="--enable-dataplane-v2 --enable-ip-alias --enable-multi-networking --network=${NETWORK_NAME_1} --subnetwork=${NETWORK_NAME_1}"

export NODE_POOL_ARGUMENTS="--additional-node-network network=${NETWORK_NAME_2},subnetwork=${SUBNET_NAME_2}"

python3 xpk.py cluster create --cluster $CLUSTER_NAME --cluster-cpu-machine-type=n1-standard-8 --num-slices=$NUM_SLICES --tpu-type=$TPU_TYPE --zone=$ZONE  --project=$PROJECT --on-demand --custom-cluster-arguments="${CLUSTER_ARGUMENTS}" --custom-nodepool-arguments="${NODE_POOL_ARGUMENTS}" --create-vertex-tensorboard --gke-version=1.30.5-gke.1014000

kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/ai-on-gke/9ff340f07f70be0130454f9e7238551587242b75/scripts/network-setup/v6e-network-optimization.yaml

gcloud container clusters upgrade $CLUSTER_NAME --master --cluster-version 1.30.3-gke.1969001 --project $PROJECT --zone $ZONE
