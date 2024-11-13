# Instruction for StableDiffusion MLPerf workload

## 1. Problem

Stable Diffusion v2 Model

### Requirements

*   [Docker](https://cloud.google.com/artifact-registry/docs/docker/authentication)
*   [GKE (Google Kubernetes Engine)](https://cloud.google.com/kubernetes-engine)
*   [XPK permission requirement](https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#cloud-console-permissions-on-the-user-or-service-account-needed-to-run-xpk)
* [ XPK prerequisites](https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#prerequisites)
* Request google cloud project and compute access and config:
```
export PROJECT=#<your_project_id>
export ZONE=#<zone>
gcloud config set project $PROJECT
gcloud config set compute/zone $ZONE
```
* create your GCS bucket for output artifacts
```
OUTPUT_DIR=gs://your_bucket_name #<your_GCS_folder_for_results>
gcloud storage buckets create ${OUTPUT_DIR}  --project ${PROJECT}
```

## 2. Directions
### Dependency Setup
Build docker image based on `maxdiffusion_fork/maxdiffusion.Dockerfile`

prebuilt-docker [here](https://pantheon.corp.google.com/artifacts/docker/cloud-tpu-multipod-dev/us/gcr.io/maxdiffusion_base_1004_exp?cloudshell=true&e=13803378&mods=allow_workbench_image_override&project=cloud-tpu-multipod-dev&supportedpurview=project) request to access

```
LOCAL_IMAGE_NAME=maxdiffusion_base_1004_exp
PROJECT=your_project

docker build  --no-cache --network host -f ./maxdiffusion.Dockerfile -t ${LOCAL_IMAGE_NAME} .

docker tag ${LOCAL_IMAGE_NAME} gcr.io/$PROJECT/${LOCAL_IMAGE_NAME}:latest
docker push gcr.io/$PROJECT/${LOCAL_IMAGE_NAME}:latest
```
#### Cluster Setup
We use [xpk](https://github.com/google/xpk) to create cluster for v5p-16, v5p-128, v5p-1024, v5p-2048 seperately.
Follow this https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#installation to install xpk

refer to setup.sh for more details. The scripts are sanitized to protect our compute project/zone information and users need to specify computer configs (GCS name, compute project, compute zone, capacity type) for their own case.

```
CLUSTER_NAME=mlperf-${TPU_TYPE}

# network setup
gcloud compute networks create "${NETWORK_NAME}" --mtu=8896 --project="${PROJECT}" --subnet-mode=auto --bgp-routing-mode=regional
gcloud compute firewall-rules create "${NETWORK_FW_NAME}" --network "${NETWORK_NAME}" --allow tcp,icmp,udp --project="${PROJECT}"

export CLUSTER_ARGUMENTS="--network=${NETWORK_NAME} --subnetwork=${NETWORK_NAME}"
python3 xpk.py cluster create --cluster "${CLUSTER_NAME}" \
  --num-slices="${NUM_SLICES}" --tpu-type="${TPU_TYPE}" --zone="${ZONE}" \
  --project="${PROJECT}" --on-demand \
  --custom-cluster-arguments="${CLUSTER_ARGUMENTS}"
```

repeat the cluster setup for v5p-16, v5p-128, v5p-1024, v5p-2048

### Steps to launch training
We use [xpk](https://github.com/google/xpk) to deploy jobs as well.

```
bash v5p-16.sh
bash v5p-128.sh
bash v5p-1024.sh
bash v5p-2048.sh
```

## 3. Dataset

Please refer to the
[instructions](https://github.com/mlcommons/training/tree/master/stable_diffusion)
from the reference to download the dataset.

We preprocessed the data:
#### 1. Download the dataset as instructed [here](https://github.com/mlcommons/training/tree/master/stable_diffusion#the-datasets)
#### 2. Create a persistent disk and attach to VM as read-write.
#### 3. Create 2 directories inside the persistent disk to store the extracted files and created tfrecords.
#### 4. [Optional] request access to the bucket by providing google compute account `gs://mlperf-llm-public2/`
#### 5. Or download corresponding ckpt from
##### 5.1 
download inception_weights_path: https://www.dropbox.com/s/xt6zvlvt22dcwck/inception_v3_weights_fid.pickle?dl=1
##### `gsutil -m cp inception_v3_weights_fid.pickle ${bucket_name}/sd/ `
##### 5.2 
download clip_model_name_or_path: https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main
##### `gsutil -m cp CLIP-ViT-H-14-laion2B-s32B-b79K ${bucket_name}/sd `

##### 5.3 
download stable_base: https://huggingface.co/stabilityai/stable-diffusion-2-base/tree/main
##### `gsutil -m cp models--stabilityai--stable-diffusion-2-base ${bucket_name}/sd `

#### 6. Run this [file](https://github.com/AI-Hypercomputer/maxdiffusion/blob/mlperf_4/src/maxdiffusion/pedagogical_examples/to_tfrecords.py) to preprocess and pre-encode text embedding:
```
python to_tfrecords.py \
  src/maxdiffusion/configs/base_2_base.yml attention=dot_product \
  data_files_pattern=/mnt/data/webdataset-moments-filtered/*.tar \
  extracted_files_dir=/tmp/raw-data-extracted \
  tfrecords_dir=/mnt/data/tf_records_512_encoder_state_fp32 \
  pretrained_model_name_or_path=gs://$bucket_name/sd/models--stabilityai--stable-diffusion-2-base \
  inception_weights_path=gs://$bucket_name/sd/inception_v3_weights_fid.pickle \
  clip_model_name_or_path=gs://$bucket_name/sd/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K \
  train_data_dir=/tmp/ \
  run_name=test no_records_per_shard=12720 base_output_directory=/tmp/output 
```
#### 7. uploaded to your gcs bucket location `gs://${bucket_name}/laion400m/raw_data/tf_records_512_encoder_state_fp32`

``` gsutil -m cp -r /mnt/data/tf_records_512_encoder_state_fp32 gs://${bucket_name}/laion400m/raw_data ```

## 4. Model

The model largely follows the Stable Diffusion v2 reference paper, with key
model architecture configs refer
[here](https://github.com/mlcommons/training/tree/master/stable_diffusion#the-model)

### List of Layers

The model largely follows the Stable Diffusion v2 reference paper. refer to
[here](https://github.com/mlcommons/training/tree/master/stable_diffusion#the-model)

## 5. Quality

### Evaluation frequency

Every 512,000 images, or CEIL(512000 / global_batch_size) if 512,000 is not
divisible by GBS.

### Evaluation thoroughness

Evaluation on the validation subset that consists of 30,000 examples subset of
coco-2014. ref to
[here](https://github.com/mlcommons/training/tree/master/stable_diffusion#evaluation-thoroughness)


## 6. Additional notes

####Postprocess for MLLOG from raw run

```
cat ${job_dir}/sd_worker.log | grep MLLOG  > ${job_dir}/result_0.txt
```

####Padded coco 30k val dataset for distributed evaluation

val2014_30k.tsv is padded with last element from 30,000 to 30,720 for
evenly-distributed loading. However filled elements are discarded during eval.