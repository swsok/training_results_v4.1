This directory contains files that are needed to run `stable_diffusion` benchmark on 
`PRIMERGY_CDI_H100_NVL_94GBx16` environment.

- `config_common.sh`                config file
- `config_PGCDI_01x16x64.sh`        config file
- `do_sd_16_all.sh`                 benchmark running script
- `do_eval_sd.sh`                   benchmark running script
- `password.txt`                    password for sudo command
- `README.md`                       this file
- `run.sub`                         slurm batch file
- `run_eval.sub`                    slurm batch file

Run the following to get the benchmark results.
1) Prepare benchmark environments according to ../pytorch/README.md, 
   for example, benchmark dataset or model, etc.

2) Fix configs, slurm batch file or benchmark running script as you need.

3) Run `do_sd_16_all.sh` to get benchmark score.

4) Run `do_eval_sd.sh` to get evaluation score. Set `CKPT_PATH` to the 
   checkpoint path.
