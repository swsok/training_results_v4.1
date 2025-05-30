# Instruction for GPT3 MLPerf workload

## 1. Problem

Large Language Model - GPT3 175B

### Requirements

*   [Docker](https://cloud.google.com/artifact-registry/docs/docker/authentication)
*   [GKE (Google Kubernetes Engine)](https://cloud.google.com/kubernetes-engine)
*   [XPK](https://github.com/AI-Hypercomputer/xpk/blob/main/README.md)

## 2. Directions

These directions should apply across all v5p slice types we used.

### Cluster Creation

We use [xpk](https://github.com/google/xpk) to create clusters. See `setup.sh`
for example commands. Once you've configured the script with the correct
project/zone information for your organization, you can create the cluster with.

```
bash setup.sh
```

Note that we use specific topologies for our v5p slices, which may require some
changes to the commands you execute with xpk. Specifically we use:

1.  v5p-4096: 4x8x64
2.  v5p-8192: 8x16x32
3.  v5p-12288: 8x16x48

### Network Setup

Follow the
[CoreDNS setup instructions](https://github.com/AI-Hypercomputer/xpk/blob/fba388fcf1817eca27e4d84301dd0a06a20a6311/xpk-large-scale-guide.sh#L187-L269)
to configure the network for your cluster.

**Note**: You do not need to do cluster creation with any special arguments as
shown in
[this section](https://github.com/AI-Hypercomputer/xpk/blob/fba388fcf1817eca27e4d84301dd0a06a20a6311/xpk-large-scale-guide.sh#L261-L266).
You can instead re-run the commands you used in the `Cluster Creation` section
above.

### Dataset

Please refer to the
[instructions](https://github.com/mlcommons/training/blob/master/large_language_model/paxml/README.md)
from the reference to download the dataset.

The C4 dataset location: `gs://mlperf-llm-public2/c4`

The tokenizer location as the SPM variable are: `gs://mlperf-llm-public2/vocab`.

#### Dataset Split

There are 1024 tfrecords in the original train split
`gs://mlperf-llm-public2/c4/en/3.0.4/` which didn't match 1536 hosts in large
scale v5p-12288 run.

We simply split each tfrecord file into 6 and create a new split containing 6144
tfrecords to address the above issue. Additionally, `dataset_info.json` should
be updated accordingly.

```
cd ../data_scripts

# change to your gcs folder in batch_split_tfrecords.sh
bash batch_split_tfrecords.sh

# a gcs folder gs://some-bucket/some-dataset-path/c4/en/3.0.7/ as an example
python create_new_shard_info.py --gcs-prefix=gs://some-bucket/some-dataset-path/c4/en/3.0.7/
```

### Steps to launch training

We use [xpk](https://github.com/google/xpk) to deploy jobs as well. We can
utilize commands similar to those in `run_and_time.sh`, such as ```
WORKLOAD_NAME=${USER}-v5p-4096 DEVICE_TYPE=v5p-4096 SCRIPT=v5p-4096.sh bash
xpk_run.sh

WORKLOAD_NAME=${USER}-v5p-8192 DEVICE_TYPE=v5p-8192 SCRIPT=v5p-8192.sh bash
xpk_run.sh

WORKLOAD_NAME=${USER}-v5p-12288 DEVICE_TYPE=v5p-12288 SCRIPT=v5p-12288.sh bash
xpk_run.sh ```

The `SCRIPT` will be attached as the workload inside each pod. And we use
`xpk_run.sh` to trigger the work deployment on a cluster.

Each `SCRIPT` covers both gpt3 task running and timing at the end. Take a look
at each of the scripts to see the configurable arguments (i.e. bucket names, run
names, etc.).

## 3. Model

The model largely follows the GPT-3 paper, with key model architecture configs
refer
[here](https://github.com/mlcommons/training/blob/master/large_language_model/paxml/README.md#3-model)

### List of Layers

The model largely follows the GPT3 [paper](https://arxiv.org/abs/2005.14165),
refer
[here](https://github.com/mlcommons/training/blob/master/large_language_model/paxml/README.md#3-model)
for model details.

### Model checkpoint

In the benchmarking region, we convert from a reference pax checkpoint which is
trained with Global Batch Size of 1536 for 4000 iterations.

To resume training, firstly the checkpoint needs to be converted from the
[Paxml](https://github.com/google/paxml) reference checkpoint to
[maxtext](https://github.com/google/maxtext/tree/main) one by running

```
WORKLOAD_NAME=${USER}-ckpt-convert SCRIPT=ckpt-convert.sh bash xpk_run.sh
```

See
[`convert_gpt3_ckpt_from_paxml.py`](https://github.com/AI-Hypercomputer/maxtext/blob/a6b44b3a40ff3ce5ed0b1d2d0595305840ed8a64/MaxText/convert_gpt3_ckpt_from_paxml.py)
for detailed conversion.

## 4. Quality

### Quality metric

Log Perplexity

### Quality target

2.69

### Evaluation frequency

Evaluate after every 24576 sequences with a length of 2048 each (=50.33B tokens)

### Evaluation thoroughness

Evaluation on the validation subset that consists of 24567 examples.

## 5. Additional notes

postprocess for MLLOG from raw run

```
cat ${job_dir}/large_scale_multislice_test_log | grep MLLOG  > ${job_dir}/result_0.txt
```
