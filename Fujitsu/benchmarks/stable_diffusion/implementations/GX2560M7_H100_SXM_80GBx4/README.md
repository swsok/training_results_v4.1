This directory contains files that are needed to run `stable_diffusion` benchmark on 
`GX2560M7_H100_SXM_80GBx4` environment.

- `config_GX2560M7_01x04x64.sh`     config file
- `config_GX2560M7_common.sh`       config file
- `README.md`                       this file

Run the following to get the benchmark results.
1) Prepare benchmark environments according to ../pytorch/README.md, 
   for example, copy run.sub file from ../pytorch.

2) Fix configs, slurm batch file or benchmark running script as you need.

3) Run the following commands:
```bash
export DATADIR="/path/to/your/datasets"
export CHECKPOINTS="/path/to/your/checkpoints"
export NEMOLOGS="/path/to/your/nemologs"  # needed for intermediate results between training and evaluation
export LOGDIR="/path/to/your/logdir"  # needed to save mlperf results (output logs)
export CONT=<docker/registry>/mlperf-nvidia:stable_diffusion-pyt
source config_GX2560M7_01x04x64.sh  # select config and source it
sbatch -N $DGXNNODES -t $WALLTIME run.sub  # you may be required to set --account and --partition here
```

