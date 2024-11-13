## Steps to launch training

### QuantaGrid D74H-7U

Launch configuration and system-specific hyperparameters for the QuantaGrid D74H-7U
submission are in the `../<implementation>/pytorch/config_D74H-7U.sh` script.

Steps required to launch training on QuantaGrid D74H-7U.

1. Build the docker container and push to a docker registry

```
cd ../pytorch
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Transfer the docker image to enroot container image

```
enroot import -o <path_to_enroot_image_name>.sqsh dockerd://<docker/registry:benchmark-tag>
```

3. Launch the training
```
export DATADIR_PHASE2=<path/to/packed_data>
export EVALDIR=<path/to/hdf5/eval_varlength>
export CHECKPOINTDIR_PHASE1=<path/to/ckpt/dir>
export LOGDIR=<path/to/output/dir>
export CONT="<path_to_enroot_image_name>.sqsh"
source config_D74H-7U.sh 
NEXP=10 sbatch --gpus=${DGXNGPU} -N ${DGXNNODES} run.sub

