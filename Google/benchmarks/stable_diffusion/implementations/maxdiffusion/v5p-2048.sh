#!/bin/bash
export PROJECT_ID=${PROJECT}
export ZONE=${ZONE}
TPU_TYPE=v5p-2048
NUM_DEVICES=1024

CLUSTER_NAME=mlperf-${TPU_TYPE}

gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}

uuid=$(uuidgen)

MAX_TRAIN_STEPS=2000
PER_DEVICE_BATCH_SIZE=2
LR=0.0003548
WARM_UP=0.3375
SEED=1334
INIT_LR=0

python3 ~/xpk/xpk.py  workload create --cluster  $CLUSTER_NAME  --workload "$USER"-maxdiffusion-"${uuid:0:8}"  --command "USER=$USER MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS} PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE} LR=${LR} WARM_UP=${WARM_UP} SEED=${SEED} INIT_LR=${INIT_LR} \
RUN_NAME=mlperf_${TPU_TYPE}_${uuid:0:8} METRICS_INTERVAL=250 bash run.sh"  \
--base-docker-image=gcr.io/cloud-tpu-multipod-dev/maxdiffusion_base_1004_exp:latest \
--tpu-type=${TPU_TYPE} --num-slices=1 --zone=$ZONE --project=$PROJECT_ID