This directory contains files that are needed to run bert benchmark on 
`GX2560M7_H100_SXM_80GBx4` environment.

- `config_GX2560M7_common.sh`         config file
- `config_GX2560M7_1x4x112x1_pack.sh` config file
- `README.md`                         this file

Run the following to get the benchmark results.
1) Prepare benchmark environments according to ../pytorch/README.md, 
   for example, copy run.sub file from ../pytorch.

2) Fix configs, slurm batch file as you need.

3) The launch command structure:
```bash
export EVALDIR="/path/to/your/data/hdf5/eval_varlength"
export DATADIR_PHASE2="/path/to/your/data/hdf5/training-4320/hdf5_4320_shards_varlength_shuffled"
export DATADIR_PHASE2_PACKED="/path/to/your/data/packed_data"
export CHECKPOINTDIR_PHASE1="/path/to/your/data/phase1"
export LOGDIR=</path/to/output/dir> # set the place where the output logs will be saved
export CONT=<docker/registry>/mlperf-nvidia:language_model-pyt
export CONTAINER_PRELOAD_LUSTRE=0
source config_GX2560M7_1x4x112x1_pack.sh  # select config and source it
sbatch -N ${DGXNNODES} --time=${WALLTIME} run.sub  # you may be required to set --account and --partition here
```

Replace `/path/to/your/data` with your existing path.

