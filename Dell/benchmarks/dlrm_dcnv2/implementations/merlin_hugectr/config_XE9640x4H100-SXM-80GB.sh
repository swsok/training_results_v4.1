## DL params
export RUN_SCRIPT="train.py"
export BATCHSIZE=28000
export BATCHSIZE_EVAL=131072
export LEARNING_RATE=0.004
export USE_MIXED_PRECISION=true
export SCALER=16348
export SHARDING_PLAN=auto
export MEM_COMM_BW_RATIO=7
export GEN_LOSS_SUMMARY=true
export MINIMUM_TRAINING_TIME=10
export DP_SHARDING_THRESHOLD=0.008

## System run parms
export DGXNNODES=1
export DGXNGPU=4
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=UNLIMITED

## Set clocks and walltime for maxQ and minEDP runs
if [[ "${SET_MAXQ_CLK:-0}" == "1" ]]; then
	  export MAXQ_CLK=1320
	    WALLTIME_MINUTES=$(expr ${WALLTIME_MINUTES} + ${WALLTIME_MINUTES} / 2) # 50% longer walltime
    elif [[ "${SET_MINEDP_CLK:-0}" == "1" ]]; then
	      export MINEDP_CLK=1665
	        WALLTIME_MINUTES=$(expr ${WALLTIME_MINUTES} + ${WALLTIME_MINUTES} / 3) # 33% longer walltime
fi
export WALLTIME=$((${NEXP:-1} * ${WALLTIME_MINUTES} + 5 ))

