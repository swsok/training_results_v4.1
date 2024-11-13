This directory contains files that are needed to run ssd benchmark on 
`PRIMERGY_CDI_H100_NVL_94GBx16` environment.

- `config_PGCDI_001x16x016.sh`      config file
- `do_ssd_16.sh`                    benchmark running script
- `password.txt`                    password for sudo command
- `README.md`                       this file
- `run.sub`                         slurm batch file

Run the following to get the benchmark results.
1) Prepare benchmark environments according to ../pytorch/README.md, 
   for example, benchmark dataset or model, etc.

2) Fix configs, slurm batch file or benchmark running script as you need.

3) Run `do_bert.sh`
