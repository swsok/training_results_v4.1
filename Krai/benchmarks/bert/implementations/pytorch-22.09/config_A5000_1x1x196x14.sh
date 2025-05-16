## DL params
#export BATCHSIZE=224
export BATCHSIZE=196
export GRADIENT_STEPS=14
export LR=3.7e-4
export MAX_SAMPLES_TERMINATION=20000000
export MAX_STEPS=9000
export OPT_LAMB_BETA_1=0.9
export OPT_LAMB_BETA_2=0.999
export START_WARMUP_STEP=0
export WARMUP_PROPORTION=0.0

#export EXTRA_PARAMS="--dense_seq_output --unpad --exchange_padding --dwu-group-size=2 --fused_bias_fc --fused_bias_mha --fused_dropout_add "
export EXTRA_PARAMS="--dense_seq_output --unpad --exchange_padding --fused_bias_fc --fused_bias_mha --fused_dropout_add "
export PHASE=2
export EVAL_ITER_START_SAMPLES=150000
export EVAL_ITER_SAMPLES=150000

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=04:00:00

## System config params
export DGXNGPU=1
export DGXSOCKETCORES=10
export DGXNSOCKET=1
export DGXHT=2         # HT is on is 2, HT off is 1

export CONT=swsok/mlperf-krai-bert:language_model
export BERTDIR="/home/swsok/mlperf/bert"
export DATADIR="${BERTDIR}/hdf5/training-4320/hdf5_4320_shards_varlength"
export DATADIR_PHASE2="${BERTDIR}/hdf5/training-4320/hdf5_4320_shards_varlength"
export EVALDIR="${BERTDIR}/hdf5/eval_varlength"
export CHECKPOINTDIR_PHASE1="${BERTDIR}/phase1"
export CHECKPOINTDIR="${BERTDIR}/checkpoints"
export UNITTESTDIR="${BERTDIR}/unit_test"

#export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_VISIBLE_DEVICES="0"
export NEXP=1
export OMP_NUM_THREADS=1
