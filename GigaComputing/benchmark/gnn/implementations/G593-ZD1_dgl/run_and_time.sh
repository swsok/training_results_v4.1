#!/bin/bash

cd ../dgl
export DATA_DIR=<data_path>/full # set correct <data_path>
export GRAPH_DIR=<graph_path>/full # set correct <graph_path>
export LOGDIR=</path/to/your/logdir> #set the place where the output logs will be saved
export CONT=./nvcr.io/nvdlfwea/mlperfv41/gnn.dgl
export FP8_EMBEDDING=1 # set to 1 if you want to use FP8 features instead
source config_G593-ZD1_1x8x2048.sh # use appropriate config
sbatch -N $DGXNNODES run.sub
