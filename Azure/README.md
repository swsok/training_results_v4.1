# MLPerf v4.1 Azure Submission

This is a repository of Azure's submission to the MLPerf v4.1 benchmark.  It
includes implementations of the benchmark LLAMA2 70B (Lora) optimized for running on Azure
ND H200 v5 virtual machines.  The reference implementations can be found elsewhere:
https://github.com/mlcommons/training.git

# v4.1 release

This readme was generated in October 2024, for the v4.1 round of MLPerf.

# Contents

The `benchmarks` subdirectory provides the following:
 
* Code that implements the model with the nemo framework.
* A Dockerfile which can be used to build a container for the benchmark.
* Documentation on the dataset, model, and machine setup.

# Running Benchmarks

The benchmark has been tested on the following machine configuration:

* A cluster of Azure ND H200 v5 virtual machines.
* The required software stack includes Slurm, with Enroot for running
  containers and the Slurm Pyxis plugin

Generally, a benchmark can be run with the following steps:

1. Follow the instructions in the README to download and format the input data and any required checkpoints.
2. Build the Dockerfile
3. Source the appropriate `config_*.sh` file.
4. `sbatch -N $DGXNNODES -t $WALLTIME run.sub`
