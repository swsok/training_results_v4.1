# Running mlcommon BERT benchmark on Lambda 1-Click Clusters

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
# From any worker node (faster to build)
export HEADNODE_HOSTNAME=ml-64-head-001
docker build --build-arg CACHEBUST=$(date +%s) -t $HEADNODE_HOSTNAME:5000/local/mlperf-nvidia-bert:latest .
docker push $HEADNODE_HOSTNAME:5000/local/mlperf-nvidia-bert:latest

# Verify if the image has been pushed to the registry
curl http://$HEADNODE_HOSTNAME:5000/v2/_catalog
```

Notice our Dockerfile installs `scipy 1.11.4`, otherwise `optimize.nnls` complains.
https://github.com/scipy/scipy/issues/20813

## Prepare dataset

```
export HEADNODE_HOSTNAME=$(hostname)
export DATAPATH=/home/ubuntu/ml-1cc/data/mlperf/bert
sudo mkdir -p $DATAPATH
sudo chmod -R 777 $DATAPATH

sbatch  --export=HEADNODE_HOSTNAME=$HEADNODE_HOSTNAME,DATAPATH=$DATAPATH dataset.sub
```

It took ~48 hours to get all data prepared. (24 hours spent on package the dataset). The following folders should be created: 
```
ubuntu@head1:~/ml-1cc/data/mlperf/bert$ ls -la
total 0
drwxrwxrwx 2 root   root   4096 Jul  1 14:14 .
drwxrwxrwx 2 root   ubuntu 4096 Jun 30 15:11 ..
drwxrwxrwx 2 ubuntu ubuntu 4096 Jun 30 15:19 download
drwxrwxrwx 2 ubuntu ubuntu 4096 Jun 17 14:21 hdf5
drwxrwxrwx 2 ubuntu ubuntu 4096 Jul  1 14:04 packed_data
drwxrwxrwx 2 ubuntu ubuntu 4096 Jun 30 15:01 per_seqlen
drwxrwxrwx 2 ubuntu ubuntu 4096 Jun 17 22:44 per_seqlen_parts
drwxrwxrwx 2 ubuntu ubuntu 4096 Jun 30 15:24 phase1
```


## Run training

```
# Single node
export HEADNODE_HOSTNAME=$(hostname) && \
source ./config_1cc_1x8x48x1_pack.sh && \
sbatch -N1 --ntasks-per-node=8 --gres=gpu:8 run_1cc.sub

# 2x nodes
export HEADNODE_HOSTNAME=$(hostname) && \
source ./config_1cc_2x8x24x1_pack.sh && \
sbatch -N2 --ntasks-per-node=8 --gres=gpu:8 run_1cc.sub

# 4x nodes
export HEADNODE_HOSTNAME=$(hostname) && \
source ./config_1cc_4x8x12x1_pack.sh && \
sbatch -N4 --ntasks-per-node=8 --gres=gpu:8 run_1cc.sub

# 8x nodes
export HEADNODE_HOSTNAME=$(hostname) && \
source ./config_1cc_8x8x36x1_pack.sh && \
sbatch -N8 --ntasks-per-node=8 --gres=gpu:8 run_1cc.sub
```

You should see training finished with log like this
```
 0: :::MLLOG {"namespace": "", "time_ms": 1719843792116, "event_type": "POINT_IN_TIME", "key": "tracked_stats", "value": {"throughput": 14214.65962521748}, "metadata": {"file": "/workspace/bert/run_pretraining.py", "lineno": 1850, "epoch_num": 799502}}
 0: :::MLLOG {"namespace": "", "time_ms": 1719843792116, "event_type": "INTERVAL_START", "key": "eval_start", "value": 799502, "metadata": {"file": "/workspace/bert/run_pretraining.py", "lineno": 1856, "epoch_num": 799502}}
 0: :::MLLOG {"namespace": "", "time_ms": 1719843792800, "event_type": "INTERVAL_END", "key": "eval_stop", "value": 799502, "metadata": {"file": "/workspace/bert/run_pretraining.py", "lineno": 1862, "epoch_num": 799502}}
 0: :::MLLOG {"namespace": "", "time_ms": 1719843792800, "event_type": "POINT_IN_TIME", "key": "eval_accuracy", "value": 0.6973460912704468, "metadata": {"file": "/workspace/bert/run_pretraining.py", "lineno": 1866, "epoch_num": 799502}}
 0: {'global_steps': 695, 'eval_loss': 1.7467454671859741, 'eval_mlm_accuracy': 0.6973460912704468}
 0: Training runs 1.3611578543980916 mins sustained_training_time 0
 0: (1, 740.0) {'final_loss': 0.0}
 0: :::MLLOG {"namespace": "", "time_ms": 1719843796271, "event_type": "INTERVAL_END", "key": "block_stop", "value": null, "metadata": {"file": "/workspace/bert/run_pretraining.py", "lineno": 1985, "first_epoch_num": 1}}
 0: :::MLLOG {"namespace": "", "time_ms": 1719843796271, "event_type": "INTERVAL_END", "key": "epoch_stop", "value": null, "metadata": {"file": "/workspace/bert/run_pretraining.py", "lineno": 1988, "epoch_num": 799502}}
 0: :::MLLOG {"namespace": "", "time_ms": 1719843796271, "event_type": "POINT_IN_TIME", "key": "train_samples", "value": 799502, "metadata": {"file": "/workspace/bert/run_pretraining.py", "lineno": 1990}}
 0: :::MLLOG {"namespace": "", "time_ms": 1719843796271, "event_type": "POINT_IN_TIME", "key": "eval_samples", "value": 10000, "metadata": {"file": "/workspace/bert/run_pretraining.py", "lineno": 1993}}
 0: :::MLLOG {"namespace": "", "time_ms": 1719843796272, "event_type": "INTERVAL_END", "key": "run_stop", "value": null, "metadata": {"file": "/workspace/bert/run_pretraining.py", "lineno": 1996, "status": "aborted"}}
 0: :::MLLOG {"namespace": "", "time_ms": 1719843796272, "event_type": "POINT_IN_TIME", "key": "tracked_stats", "value": {"throughput": 12792.219630021962, "epoch_num": 799502}, "metadata": {"file": "/workspace/bert/run_pretraining.py", "lineno": 2035, "step": [2, 740]}}
 0: {'e2e_time': 126.1371500492096, 'training_sequences_per_second': 10438.167147016668, 'final_loss': 0.0, 'raw_train_time': 81.66951036453247}
```