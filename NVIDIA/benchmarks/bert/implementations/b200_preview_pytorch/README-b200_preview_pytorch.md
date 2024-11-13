## Steps to launch training

### b200_preview_pytorch

Launch configuration and system-specific hyperparameters for the
b200_preview_pytorch submission are in the
`benchmarks/bert/implementations/b200_preview_pytorch/config_DGXB200_1x8x48x1_pack.sh` script.

Steps required to launch training for b200_preview_pytorch.  The sbatch
script assumes a cluster running Slurm with the Pyxis containerization plugin.

1. Build the docker container and push to a docker registry

```
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_DGXB200_1x8x48x1_pack.sh
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N ${DGXNNODES} -t ${WALLTIME} run.sub
```
