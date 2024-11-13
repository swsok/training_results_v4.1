This directory contains files that are needed to run `llama2_70b_lora` benchmark on 
`GX2560M7_H100_SXM_80GBx4` environment.

- `configs/config_GX2560M7_1x4x8xtp4pp1cp1.sh` config file
- `configs/config_GX2560M7_common.sh`          config file
- `README.md`                                  this file

Run the following to get the benchmark results.
1) Prepare benchmark environments according to ../pytorch/README.md, 
   for example, copy run.sub file from ../pytorch.

2) Fix configs, slurm batch file as you need.

3) The launch command structure:
```bash
export DATADIR=</path/to/dataset>/gov_report # set correct </path/to/dataset>
export MODEL=</path/to/dataset>/model # set correct </path/to/dataset>
export LOGDIR=</path/to/output/dir> # set the place where the output logs will be saved
export CONT=<docker/registry>/mlperf-nvidia:llama2_70b_lora-pyt
source configs/config_GX2560M7_1x4x8xtp4pp1cp1.sh  # select config and source it
sbatch -N $DGXNNODES -t $WALLTIME run.sub  # you may be required to set --account and --partition here
```

Replace `/path/to/your/data` with your existing path.
