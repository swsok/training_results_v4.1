# Running mlcommon Llama2_70b_lora benchmark on Lambda 1-Click Clusters

## Install SLURM/Enroot/Pyxis/Docker registry
These are pre-installed by Lambda engineers. 

You should be able to submit SLURM job from the head node
```
ubuntu@head1:~$ sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
batch*       up   infinite      2   idle worker[1-2]
```

A docker registry should be running on the head node
```
ubuntu@head1:~$ docker ps
CONTAINER ID   IMAGE          COMMAND                  CREATED      STATUS       PORTS     NAMES
7ac6b15b8518   registry:2.8   "/entrypoint.sh /etc…"   5 days ago   Up 3 hours             deepops-registry
``` 

### Docker Configuration

On ALL nodes, remove the need of `sudo` for running docker
```
export USERNAME=$(whoami)
sudo groupadd docker
sudo usermod -aG docker $USERNAME
newgrp docker
```

## Buid Docker Container

```
# Build the container and push to local registry
# Currently head node will crash during docker build, so better use a worker node to build the image and push to the head node registery
export HEADNODE_HOSTNAME=ml-64-head-001
docker build --build-arg CACHEBUST=$(date +%s) -t $HEADNODE_HOSTNAME:5000/local/mlperf-nvidia-llama2_70b_lora:latest .
docker push $HEADNODE_HOSTNAME:5000/local/mlperf-nvidia-llama2_70b_lora:latest

# Verify if the image has been pushed to the registry
curl http://$HEADNODE_HOSTNAME:5000/v2/_catalog
```

## Prepare dataset

```
# see if scripts are in the right place
docker run -it $HEADNODE_HOSTNAME:5000/local/mlperf-nvidia-llama2_70b_lora:latest
```

```
export HEADNODE_HOSTNAME=$(hostname)
export DATAPATH=/home/ubuntu/ml-1cc/data/mlperf/llama2_70b_lora/data
export MODELPATH=/home/ubuntu/ml-1cc/data/mlperf/llama2_70b_lora/ckpt
sudo mkdir -p $DATAPATH
sudo chmod -R 777 $DATAPATH
sudo mkdir -p $MODELPATH
sudo chmod -R 777 $MODELPATH
sbatch -N1 --ntasks-per-node=1 --export=HEADNODE_HOSTNAME=$HEADNODE_HOSTNAME,DATAPATH=$DATAPATH,MODELPATH=$MODELPATH dataset.sub
```

Once done you should see following folders:

```
ubuntu@ml-512-head-002:~/ml-1cc/data/mlperf$ tree llama2_70b_lora/ -L 1
llama2_70b_lora/
├── ckpt
└── data
```

## Run training

```
# Single node
export HEADNODE_HOSTNAME=$(hostname) && \
source configs/config_1cc_1x8x4xtp4pp1cp1.sh && \
sbatch -N1 --ntasks-per-node=8 --gres=gpu:8 run_1cc.sub


export HEADNODE_HOSTNAME=$(hostname) && \
source configs/config_1cc_1x8x4xtp4pp1cp1_1008.sh && \
sbatch -N1 --ntasks-per-node=8 --gres=gpu:8 run_1cc.sub

# 2x nodes
export HEADNODE_HOSTNAME=$(hostname) && \
source configs/config_1cc_2x8x2xtp4pp1cp1.sh && \
sbatch -N2 --ntasks-per-node=8 --gres=gpu:8 run_1cc.sub

# 4x nodes
export HEADNODE_HOSTNAME=$(hostname) && \
source configs/config_1cc_4x8x1xtp4pp1cp1.sh && \
sbatch -N4 --ntasks-per-node=8 --gres=gpu:8 run_1cc.sub


# 8x nodes
export HEADNODE_HOSTNAME=$(hostname) && \
source configs/config_1cc_8x8x1xtp4pp1cp1.sh && \
sbatch -N8 --ntasks-per-node=8 --gres=gpu:8 run_1cc.sub
```

You should see training finished with log like this
```
0: :::MLLOG {"namespace": "", "time_ms": 1721026606541, "event_type": "INTERVAL_START", "key": "block_start", "value": null, "metadata": {"file": "/workspace/ft-llm/custom_callbacks.py", "lineno": 129, "samples_count": 2688}}
0: :::MLLOG {"namespace": "", "time_ms": 1721026805975, "event_type": "POINT_IN_TIME", "key": "tracked_stats", "value": {"throughput": 1.9257918929116467}, "metadata": {"file": "/workspace/ft-llm/custom_callbacks.py", "lineno": 161, "step": 3072}}
0: :::MLLOG {"namespace": "", "time_ms": 1721026805975, "event_type": "INTERVAL_END", "key": "block_stop", "value": null, "metadata": {"file": "/workspace/ft-llm/custom_callbacks.py", "lineno": 112, "samples_count": 3072}}
0: :::MLLOG {"namespace": "", "time_ms": 1721026805975, "event_type": "INTERVAL_START", "key": "eval_start", "value": null, "metadata": {"file": "/workspace/ft-llm/custom_callbacks.py", "lineno": 117, "samples_count": 3072}}
0: :::MLLOG {"namespace": "", "time_ms": 1721026850352, "event_type": "POINT_IN_TIME", "key": "eval_accuracy", "value": 0.9234074354171753, "metadata": {"file": "/workspace/ft-llm/custom_callbacks.py", "lineno": 181, "samples_count": 3072}}
0: :::MLLOG {"namespace": "", "time_ms": 1721026850352, "event_type": "INTERVAL_END", "key": "eval_stop", "value": null, "metadata": {"file": "/workspace/ft-llm/custom_callbacks.py", "lineno": 186, "samples_count": 3072}}
0: :::MLLOG {"namespace": "", "time_ms": 1721026850352, "event_type": "INTERVAL_END", "key": "run_stop", "value": 0.9234074354171753, "metadata": {"file": "/workspace/ft-llm/custom_callbacks.py", "lineno": 195, "samples_count": 3072, "status": "success"}}
```