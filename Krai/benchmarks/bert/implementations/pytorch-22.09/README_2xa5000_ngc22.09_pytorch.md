## Steps to launch training on a single node with 2xA30

### NVIDIA DGX single node
Launch configuration and system-specific hyperparameters for the NVIDIA A5000
multi node submission are in the following scripts:
* for the 2xA5000 1-node NVIDIA submission: `config_A5000_1x2x224x14.sh`

Steps required to launch multi node training on NVIDIA 2xA5000:

1. Build the container:

```
docker build --pull -t <docker/registry>/mlperf-nvidia:language_model .
docker push <docker/registry>/mlperf-nvidia:language_model
```

2. Launch the training:

1-node NVIDIA 2xA5000 training:

change BERTDIR path of config_A5000_*.sh  as where your dataset places.

1 GPU training

```
source config_A5000_1x1x192x14.sh
./run_with_docker.sh
```

2 GPU training

```
source config_A5000_1x2x224x14.sh
./run_with_docker.sh
```

