This directory contains files that are needed to run ssd benchmark on 
`GX2560M7_H100_SXM_80GBx4` environment.

- `config_GX2560M7_001x04x064.sh`   config file
- `README.md`                       this file

README.md

Run the following to get the benchmark results.
1) Prepare benchmark environments according to ../pytorch/README.md, 
   for example, copy run.sub file from ../pytorch.

2) Fix configs, slurm batch file or benchmark running script as you need.

3) Run the following commands:
```bash
export DATADIR="<path/to/dir/containing/openimages/dir>"
export BACKBONE_DIR="<path/to/pretrained/ckpt>"
export LOGDIR="<path/to/output/dir>"
export CONT=<docker/registry>/mlperf-nvidia:single_stage_detector-pyt
source config_GX2560M7_001x04x064.sh  # select config and source it
sbatch -N $DGXNNODES -t $WALLTIME run.sub  # you may be required to set --account and --partition here
```

