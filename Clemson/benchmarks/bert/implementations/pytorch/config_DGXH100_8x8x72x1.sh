## DL params

export BATCHSIZE=72
export GRADIENT_STEPS=1
export PACKING_FACTOR=1
export LR=0.00258
export MAX_SAMPLES_TERMINATION=4500000
export MAX_STEPS=700
export OPT_LAMB_BETA_1=0.6
export OPT_LAMB_BETA_2=0.7
export START_WARMUP_STEP=-200000
export WARMUP_STEPS=200330
export WEIGHT_DECAY_RATE=0.1
export INIT_LOSS_SCALE=1024.0

export EXTRA_PARAMS="--dense_seq_output --pad_fmha --fused_bias_fc --fused_bias_mha --fused_dropout_add --fused_gemm_gelu --packed_samples --use_transformer_engine2 --cuda_graph_mode 'segmented' --use_cuda_graph "

export PHASE=2
export EVAL_ITER_START_SAMPLES=200000
export EVAL_ITER_SAMPLES=200000

## System run parms
export DGXNNODES=6
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME_MINUTES=4
export WALLTIME=$(( ${NEXP:-1} * ${WALLTIME_MINUTES} + 5 ))

## System config params
source $(dirname ${BASH_SOURCE[0]})/config_DGXA100_common.sh

export CONTAINER_PRELOAD_LUSTRE=1
