# MLPerf v4.1 Krai Submission

This is a repository of Krai's submission to the MLPerf v4.1 benchmark. It
includes optimized implementations of the benchmark code. The reference
implementations can be found elsewhere:
https://github.com/mlcommons/training.git

# v4.1 release

This readme was updated in October. 2024, for the v4.1 round of MLPerf.

# Contents

The implementation(s) in the `benchmarks` subdirectory provides the following:
 
* Code that implements the model in at least one framework.
* A Dockerfile which can be used to run the benchmark in a container.
* Documentation on the dataset, model, and machine setup.

# Running the Benchmark

These benchmarks have been tested on the following machine configuration:

* A server with 2x NVIDIA RTX A5000s (2x24GB gpus) using PyTorch 22.09.

Please see [here](./benchmarks/bert/implementations/pytorch-22.09/README_2xa5000_ngc22.09_pytorch.md) for the detail instructions in running the benchmark. 

