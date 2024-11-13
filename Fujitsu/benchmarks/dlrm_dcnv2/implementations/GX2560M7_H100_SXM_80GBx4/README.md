This directory contains files that are needed to run `dlrm_dcnv2` benchmark on 
`GX2560M7_H100_SXM_80GBx4` environment.

- `config_GX2560M7_1x4x49152.sh`      config file
- README.md                           config file

Run the following to get the benchmark results.
1) Prepare benchmark environments according to ../hugectr/README.md, 
   for example, copy run.sub file from ../hugectr.

2) Fix configs, slurm batch file as you need.

3) The launch command structure:
```bash
export DATADIR=/path/to/train/criteo_multihot_raw
export DATADIR_VAL=/path/to/val/criteo_multihot_raw
source config_GX2560M7_1x4x49152.sh  # select config and source it
sbatch -N $DGXNNODES -t $WALLTIME run.sub  # you may be required to set --account and --partition here
```
