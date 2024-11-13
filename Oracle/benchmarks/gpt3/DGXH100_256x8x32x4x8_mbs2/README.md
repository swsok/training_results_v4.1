## Running NVIDIA Large Language Model GPT-3 175B PyTorch MLPerf Benchmark

This file contains the instructions for running the NVIDIA Large Language Model GPT-3 175B PyTorch MLPerf Benchmark on NVIDIA hardware.

## 1. Hardware Requirements

- At least 5TB disk space is required.
- NVIDIA GPU with at least 80GB memory is strongly recommended.
- GPUs are required for dataset preparation.

## 2. Software Requirements

- Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
- [Docker](https://www.docker.com/)

## 3. Set up

### 3.1 Build the container

Replace `<docker/registry>` with your container registry and build:

```bash
docker build -t <docker/registry>/mlperf-nvidia:large_language_model-pyt
```

### 3.2 Prepare dataset

Please refer to the [instructions](https://github.com/mlcommons/training/blob/master/large_language_model/megatron-lm/README.md#3-datasetenvironment) from the reference to download the dataset.
Separate tokenizer files download can be skipped - the unpacked dataset archive already contains the required tokenizer `c4_en_301_5Mexp2_spm.model` in the `spm` directory.

The C4 dataset location (`preprocessed_c4_spm` directory) should be set as PREPROC_DATA variable and the tokenizer location (the `c4_en_301_5Mexp2_spm.model` **file**) as the SPM variable. 

Final dataset structure on disk:

```
c4  # 331G
├── preprocessed_c4_spm
│   ├── c4_en_0_c4_spm_text_document.bin  # 41G
│   ├── c4_en_0_c4_spm_text_document.idx  # 870M
│   ├── c4_en_1_c4_spm_text_document.bin  # 41G
│   ├── c4_en_1_c4_spm_text_document.idx  # 870M
│   ├── c4_en_2_c4_spm_text_document.bin  # 41G
│   ├── c4_en_2_c4_spm_text_document.idx  # 870M
│   ├── c4_en_3_c4_spm_text_document.bin  # 41G
│   ├── c4_en_3_c4_spm_text_document.idx  # 870M
│   ├── c4_en_4_c4_spm_text_document.bin  # 41G
│   ├── c4_en_4_c4_spm_text_document.idx  # 870M
│   ├── c4_en_5_c4_spm_text_document.bin  # 41G
│   ├── c4_en_5_c4_spm_text_document.idx  # 870M
│   ├── c4_en_6_c4_spm_text_document.bin  # 41G
│   ├── c4_en_6_c4_spm_text_document.idx  # 870M
│   ├── c4_en_7_c4_spm_text_document.bin  # 41G
│   ├── c4_en_7_c4_spm_text_document.idx  # 870M
│   ├── c4_en_validation_subset_c4_spm_text_document.bin  # 23M
│   └── c4_en_validation_subset_c4_spm_text_document.idx  # 480K
└── tokenizers
    └── google_c4_spm
        └── c4_en_301_5Mexp2_spm.model  #1.1M
```

### 3.3 Model and checkpoint prepration

#### 3.3.1 Publication/Attribution
[Megatron](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/nemo_megatron/intro.html) is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA. This repository uses [Nemo Megatron](https://github.com/NVIDIA/NeMo). NeMo Megatron GPT has been integrated with [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine). Transformer Engine enables FP8 training on NVIDIA Hopper GPUs.

#### 3.3.2 List of Layers

The model largely follows the GPT3 [paper](https://arxiv.org/abs/2005.14165), refer [here](https://github.com/mlcommons/training/tree/master/large_language_model/megatron-lm#list-of-layers) for model details.

#### 3.3.3 Model checkpoint
In the benchmarking region, we resume training from a reference checkpoint which is trained with Global Batch Size of 1536 for 4000 iterations. 

Please refer to the [instructions](https://github.com/mlcommons/training/blob/master/large_language_model/megatron-lm/README.md#checkpoint-download) from the reference to download the BF16 model checkpoint.
The postprocessing step can be skipped - the `gpt3/megatron-lm/checkpoint_nemo_bf16.tar` is already NeMo-compatible after unpacking.

The `LOAD_CHECKPOINTS_PATH` variable should be set to the **parent** directory of the `ckpt4000-consumed_samples=0` checkpoint.

For more details on the checkpoint format, please refer to the reference checkpoint [description](https://github.com/mlcommons/training/blob/master/large_language_model/megatron-lm/README.md#model-checkpoint). 

Between mlperf training v4.0 and v4.1 the code responsible for mapping tensor names was removed in favour of renaming directories in the checkpoint. The script used for the conversion is `convert_v4.0_checkpoint_to_v4.1.sh`. The checkpoint used for v4.0 works no longer.

Final checkpoint structure on disk:
```
dist_ckpt/
└── ckpt4000-consumed_samples=0  # 2.3T
    ├── common.pt
    ├── metadata.json
    ├── model.decoder.final_layernorm.bias
    ├── model.decoder.final_layernorm.weight
    ├── model.decoder.layers.layernorm_mlp.fc1_bias
    ├── model.decoder.layers.layernorm_mlp.fc1_weight
    ├── model.decoder.layers.layernorm_mlp.fc2_bias
    ├── model.decoder.layers.layernorm_mlp.fc2_weight
    ├── model.decoder.layers.layernorm_mlp.layer_norm_bias
    ├── model.decoder.layers.layernorm_mlp.layer_norm_weight
    ├── model.decoder.layers.self_attention.layernorm_qkv.bias
    ├── model.decoder.layers.self_attention.layernorm_qkv.layer_norm_bias
    ├── model.decoder.layers.self_attention.layernorm_qkv.layer_norm_weight
    ├── model.decoder.layers.self_attention.layernorm_qkv.weight
    ├── model.decoder.layers.self_attention.proj.bias
    ├── model.decoder.layers.self_attention.proj.weight
    ├── model.embedding.position_embeddings.weight
    ├── model.embedding.word_embeddings.weight
    ├── optimizer.state.exp_avg.model.decoder.final_layernorm.bias
    ├── optimizer.state.exp_avg.model.decoder.final_layernorm.weight
    ├── optimizer.state.exp_avg.model.decoder.layers.layernorm_mlp.fc1_bias
    ├── optimizer.state.exp_avg.model.decoder.layers.layernorm_mlp.fc1_weight
    ├── optimizer.state.exp_avg.model.decoder.layers.layernorm_mlp.fc2_bias
    ├── optimizer.state.exp_avg.model.decoder.layers.layernorm_mlp.fc2_weight
    ├── optimizer.state.exp_avg.model.decoder.layers.layernorm_mlp.layer_norm_bias
    ├── optimizer.state.exp_avg.model.decoder.layers.layernorm_mlp.layer_norm_weight
    ├── optimizer.state.exp_avg.model.decoder.layers.self_attention.layernorm_qkv.bias
    ├── optimizer.state.exp_avg.model.decoder.layers.self_attention.layernorm_qkv.layer_norm_bias
    ├── optimizer.state.exp_avg.model.decoder.layers.self_attention.layernorm_qkv.layer_norm_weight
    ├── optimizer.state.exp_avg.model.decoder.layers.self_attention.layernorm_qkv.weight
    ├── optimizer.state.exp_avg.model.decoder.layers.self_attention.proj.bias
    ├── optimizer.state.exp_avg.model.decoder.layers.self_attention.proj.weight
    ├── optimizer.state.exp_avg.model.embedding.position_embeddings.weight
    ├── optimizer.state.exp_avg.model.embedding.word_embeddings.weight
    ├── optimizer.state.exp_avg_sq.model.decoder.final_layernorm.bias
    ├── optimizer.state.exp_avg_sq.model.decoder.final_layernorm.weight
    ├── optimizer.state.exp_avg_sq.model.decoder.layers.layernorm_mlp.fc1_bias
    ├── optimizer.state.exp_avg_sq.model.decoder.layers.layernorm_mlp.fc1_weight
    ├── optimizer.state.exp_avg_sq.model.decoder.layers.layernorm_mlp.fc2_bias
    ├── optimizer.state.exp_avg_sq.model.decoder.layers.layernorm_mlp.fc2_weight
    ├── optimizer.state.exp_avg_sq.model.decoder.layers.layernorm_mlp.layer_norm_bias
    ├── optimizer.state.exp_avg_sq.model.decoder.layers.layernorm_mlp.layer_norm_weight
    ├── optimizer.state.exp_avg_sq.model.decoder.layers.self_attention.layernorm_qkv.bias
    ├── optimizer.state.exp_avg_sq.model.decoder.layers.self_attention.layernorm_qkv.layer_norm_bias
    ├── optimizer.state.exp_avg_sq.model.decoder.layers.self_attention.layernorm_qkv.layer_norm_weight
    ├── optimizer.state.exp_avg_sq.model.decoder.layers.self_attention.layernorm_qkv.weight
    ├── optimizer.state.exp_avg_sq.model.decoder.layers.self_attention.proj.bias
    ├── optimizer.state.exp_avg_sq.model.decoder.layers.self_attention.proj.weight
    ├── optimizer.state.exp_avg_sq.model.embedding.position_embeddings.weight
    ├── optimizer.state.exp_avg_sq.model.embedding.word_embeddings.weight
    ├── optimizer.state.fp32_param.model.decoder.final_layernorm.bias
    ├── optimizer.state.fp32_param.model.decoder.final_layernorm.weight
    ├── optimizer.state.fp32_param.model.decoder.layers.layernorm_mlp.fc1_bias
    ├── optimizer.state.fp32_param.model.decoder.layers.layernorm_mlp.fc1_weight
    ├── optimizer.state.fp32_param.model.decoder.layers.layernorm_mlp.fc2_bias
    ├── optimizer.state.fp32_param.model.decoder.layers.layernorm_mlp.fc2_weight
    ├── optimizer.state.fp32_param.model.decoder.layers.layernorm_mlp.layer_norm_bias
    ├── optimizer.state.fp32_param.model.decoder.layers.layernorm_mlp.layer_norm_weight
    ├── optimizer.state.fp32_param.model.decoder.layers.self_attention.layernorm_qkv.bias
    ├── optimizer.state.fp32_param.model.decoder.layers.self_attention.layernorm_qkv.layer_norm_bias
    ├── optimizer.state.fp32_param.model.decoder.layers.self_attention.layernorm_qkv.layer_norm_weight
    ├── optimizer.state.fp32_param.model.decoder.layers.self_attention.layernorm_qkv.weight
    ├── optimizer.state.fp32_param.model.decoder.layers.self_attention.proj.bias
    ├── optimizer.state.fp32_param.model.decoder.layers.self_attention.proj.weight
    ├── optimizer.state.fp32_param.model.embedding.position_embeddings.weight
    └── optimizer.state.fp32_param.model.embedding.word_embeddings.weight
```

## 4. Launch training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

Navigate to the directory where `run.sub` is stored.

The launch command structure:
```bash
export PREPROC_DATA="/path/to/your/preprocessed_c4_spm"
export SPM="/path/to/your/c4_en_301_5Mexp2_spm.model"
export LOAD_CHECKPOINTS_PATH="/path/to/your/dist_ckpt/"
export CHECKPOINT_NAME="ckpt4000-consumed_samples=0"
export LOGDIR=</path/to/output/dir>  # set the place where the output logs will be saved
export CONT=<docker/registry>/mlperf-nvidia:large_language_model-pyt
source config_DGXH100_64x8x128x4x8_mbs2.sh  # select config and source it
sbatch -N ${DGXNNODES} --time=${WALLTIME} run.sub  # you may be required to set --account and --partition here
```

Replace `/path/to/your` prefix with your existing path.

All configuration files follow the format `config_<SYSTEM_NAME>_<NODES>x<GPUS/NODE>x<BATCH/GPU>x<GRADIENT_ACCUMULATION_STEPS>.sh`.

# 5. Quality

### Quality metric
Log Perplexity

### Quality target
2.69

### Evaluation frequency
Evaluate after every 24576 samples (=50.33B tokens)

### Evaluation thoroughness
Evaluation on the validation subset that consists of 24567 examples.


# 6. Additional notes

### Config naming convention

`<number of nodes DGXNNODES>x<number of gpus per node>x<mini batch size>x<tensor parallelism TENSOR_MODEL_PARALLEL>x<pipeline parallelism PIPELINE_MODEL_PARALLEL>`

```
MP = TP * PP
DP = WS // MP
miniBS = GBS // DP
```
where: 
```
MP = model parallelism
TP = tensor parallelism
PP = pipeline parallelism
DP = data parallelism
WS = world size (number of nodes x number of gpus per node)
GBS = global batch size
```
Note: changing `MICRO_BATCH_SIZE` doesn't affect GBS or any of the above parameters.
Effectively it controls gradient accumulation (`GA = miniBS // microBS`).

Additional requirement for every config is that the GBS should be divisible by `DP*PP*MICRO_BATCH_SIZE`

### Seeds
NeMo produces dataset index shuffling only on one process and holds the `SEED` value in the file name.
Thus, all processes need to have the same value of `SEED` otherwise will not be able to read the data.
The `SEED` environment variable can be set prior to launching the job, otherwise it is set in `run.sub`.
