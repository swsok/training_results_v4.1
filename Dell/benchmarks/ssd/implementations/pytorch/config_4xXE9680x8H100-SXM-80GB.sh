#!/bin/bash

#DL params
export BATCHSIZE=${BATCHSIZE:-16}
export NUMEPOCHS=${NUMEPOCHS:-6}
export LR=${LR:-0.000085}
export WARMUP_EPOCHS=${WARMUP_EPOCHS:-0}
export EXTRA_PARAMS=${EXTRA_PARAMS:-'--jit --frozen-bn-opt --frozen-bn-fp16 --apex-adam --apex-focal-loss --apex-backbone-fusion --apex-head-fusion --disable-ddp-broadcast-buffers --reg-head-pad --cls-head-pad --cuda-graphs --dali --dali-matched-idxs --dali-eval --cuda-graphs-syn --async-coco --dali-cpu-decode --master-weights --eval-batch-size=32'}
## System run params
export DGXNNODES=4
export WALLTIME_RUNANDTIME=UNLIMITED
export WALLTIME=UNLIMITED
export MELLANOX_VISIBLE_DEVICES=0,2,3,4,5,7,8,9
export UCX_NET_DEVICES=mlx5_0:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9
## System config params
export DGXNGPU=8
export DGXSOCKETCORES=52
export DGXNSOCKET=2
export DGXHT=1  # HT is on is 2, HT off is 1
