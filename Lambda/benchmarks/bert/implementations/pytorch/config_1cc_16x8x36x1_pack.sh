## DL params                                                                                                                                                                                                          
export BATCHSIZE=36
export PACKING_FACTOR=1
export GRADIENT_STEPS=1
export LR=0.002
export MAX_SAMPLES_TERMINATION=4500000
export MAX_STEPS=740
export OPT_LAMB_BETA_1=0.6
export OPT_LAMB_BETA_2=0.7
export START_WARMUP_STEP=-200000
export WARMUP_STEPS=200330
export WEIGHT_DECAY_RATE=0.1
export INIT_LOSS_SCALE=1024.0

export SBATCH_NETWORK=sharp
export EXTRA_PARAMS="--dense_seq_output --pad_fmha --fused_bias_fc --fused_bias_mha --fused_dropout_add --fused_gemm_gelu --packed_samples --use_transformer_engine2 --cuda_graph_mode 'segmented' --use_cuda_graph "
export PHASE=2
export EVAL_ITER_START_SAMPLES=150000
export EVAL_ITER_SAMPLES=150000

## System run parms                                                                                                                                                                                                   
export DGXNNODES=8
export DGXSYSTEM="1CC"
export WALLTIME_MINUTES=4
if [[ "${MLPERF_POWER_TRAIN_AFTER_RUN_STOP:-0}" == "1" ]]; then
  export WALLTIME_MINUTES=$((${WALLTIME_MINUTES} + 15))  
  export SUSTAINED_TRAINING_TIME=11
fi
if [[ "${SET_MAXQ_CLK:-0}" == "1" ]] || [[ "${SET_MINEDP_CLK:-0}" == "1" ]]; then
  export WALLTIME_MINUTES=$((${WALLTIME_MINUTES} + 5))
  ## gpc frequency at maxQ and minEDP point
  export MAXQ_CLK=1515
  export MINEDP_CLK=1650
fi
export WALLTIME=$(( ${NEXP:-1} * ${WALLTIME_MINUTES} + 5 ))

## System config params                                                                                                                                                                                               
source $(dirname ${BASH_SOURCE[0]})/config_1cc_common.sh

export CONTAINER_PRELOAD_LUSTRE=0
export DATADIR_PHASE2=${DATADIR_PHASE2_PACKED}

## log dir
timestamp=$(date +'%y-%m-%d_%H-%M-%S')
export LOGDIR=./results/1cc_16x8x36x1_${timestamp}
mkdir -p ${LOGDIR}