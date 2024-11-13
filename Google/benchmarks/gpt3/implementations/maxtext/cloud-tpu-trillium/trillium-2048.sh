#!/bin/bash
set -euox pipefail

BRANCH=${BRANCH:-mlperf/4.1}

if [[ ! -d "maxtext" ]]; then
  git clone https://github.com/google/maxtext.git
fi

# switch branch
cd maxtext
git fetch origin "${BRANCH}"
git checkout "${BRANCH}"

sleep 60

# flags set as default

# hlo dump
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump_file"

# debug
export TPU_STDERR_LOG_LEVEL=0
export TF_CPP_MIN_LOG_LEVEL=0
export TPU_MIN_LOG_LEVEL=0

# tunable parameter
export LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-"--xla_tpu_use_megascale_host_reduction=false --xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_use_bundle_aware_cost_model_for_fusions=false"}

# checkpoint loading from a fixed folder
RUNNAME=convergence_test_0
BASE_OUTPUT_DIRECTORY=${BASE_OUTPUT_DIRECTORY:-gs://some-bucket}
DATASET_PATH=${DATASET_PATH:-gs://some-bucket/some-dataset-path}
DATASET_NAME=c4/en:3.0.7
SEED=8745

# set enable_checkpointing as true to load a checkpoint
# tunable parameters: ici_tensor_parallelism, per_device_batch_size, remat_policy, attention, int8_training
#  ici_tensor_parallelism is tunable and should be compatibility to topology
python3 MaxText/train.py MaxText/configs/base.yml run_name="${RUNNAME}" model_name=gpt3-175b\
  base_output_directory="${BASE_OUTPUT_DIRECTORY}"\
  enable_checkpointing=true async_checkpointing=false\
  per_device_batch_size=2\
  ici_fsdp_parallelism=-1\
  remat_policy=full\
  attention=flash\
  quantization=int8\
  sa_block_q=1024 sa_block_q_dkv=2048 sa_block_q_dq=2048\
  fused_qkv=false\
  tff_quant=true\
  dataset_type=c4_mlperf\
  dataset_path="${DATASET_PATH}" dataset_name="${DATASET_NAME}"\
  tokenizer_path=gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model\
  data_shuffle_seed="${SEED}"\
  2>&1 | tee /tmp/large_scale_multislice_test_log


EXP_FOLDER="${BASE_OUTPUT_DIRECTORY}/${RUNNAME}/${WORKLOAD_NAME}/${TIMESTAMP}"

if [[ ${MEGASCALE_SLICE_ID} == "0" ]]; then
  if [[ ${TPU_WORKER_ID} == "0" ]]; then
    gsutil -m cp -r /tmp/xla_dump_file "${EXP_FOLDER}/xla/"
  fi
fi

if [[ $(grep "MLLOG" /tmp/large_scale_multislice_test_log | wc -l) -gt 0 ]];then
  gsutil -m cp /tmp/large_scale_multislice_test_log "${EXP_FOLDER}/large_scale_multislice_test_log"
  bash ../parser_metrics.sh 2>&1 | tee /tmp/parser_metrics_log
  gsutil -m cp /tmp/parser_metrics_log "${EXP_FOLDER}/parser_metrics_log"
fi
