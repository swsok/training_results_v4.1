This directory contains files that are needed to run bert benchmark on 
`PRIMERGY_CDI_H100_NVL_94GBx16` environment.

- `config_DGXH100_common.sh`        config file
- `config_PGCDI_1x16x48x1_pack.sh`  config file
- `do_bert_16.sh`                   benchmark running script
- `password.txt`                    password for sudo command
- `README.md`                       this file
- `run_pretraining.py`              benchmark script that overwrite the original
- `run.sub`                         slurm batch file

Run the following to get the benchmark results.
1) Prepare benchmark environments according to ../pytorch/README.md, 
   for example, benchmark dataset or model, etc.

2) Fix configs, slurm batch file or benchmark running script as you need.

3) Run `do_bert.sh`
