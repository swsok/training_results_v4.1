#SBATCH --job-name=sd-containers
#SBATCH --gpus-per-node=8
#SBATCH --exclude=compute-permanent-node-434
#SBATCH --mem=0
#SBATCH --exclusive

CONT='/mnt/localdisk/gpt/cont/gpt3-20240923.sqsh'
MOUNT=''
echo "Running on hosts: $(echo $(scontrol show hostname))"

srun \
   --ntasks-per-node=1 \
   --ntasks=$SLURM_JOB_NUM_NODES \
   --whole --gpus-per-node=8 \
   --container-name=gpt \
   --container-image="${CONT}" \
   --container-mounts="${MOUNT}" \
   bash -c "ls -l /workspace/hpl-linux-x86_64/sample-dat;ls -l /workspace/hpl-linux-x86_64/sample-dat/HPL-dgx-a100-${SLURM_JOB_NUM_NODES}N.dat "

export PMI_DEBUG=1
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib
export OMPI_MCA_btl_tcp_if_include="10.224.0.0/16"
export PMIX_MCA_gds="^ds12" \
      NCCL_CROSS_NIC=1 \
      NCCL_SOCKET_NTHREADS=16 \
      NCCL_DEBUG=WARN \
      NCCL_CUMEM_ENABLE=0 \
      NCCL_IB_SPLIT_DATA_ON_QPS=0 \
      NCCL_IB_QPS_PER_CONNECTION=16 \
      NCCL_IB_GID_INDEX=3 \
      NCCL_IB_TC=41 \
      NCCL_IB_SL=0 \
      NCCL_IB_TIMEOUT=22 \
      NCCL_NET_PLUGIN=none \
      NCCL_SOCKET_IFNAME=eth0 \
#MPIVARS_PATH=/usr/mpi/gcc/openmpi-4.1.7a1/bin//mpivars.sh
MPIVARS_PATH=/nfs/scratch/mpi/hpcx-v2.13.1-gcc-MLNX_OFED_LINUX-5-ubuntu22.04-cuda11-gdrcopy2-nccl2.12-x86_64/hpcx-init-ompi.sh
#source $MPIVARS_PATH
#LOCAL_MPI=/usr/mpi/gcc/openmpi-4.1.7a1/bin
LOCAL_MPI=/nfs/scratch/mpi/hpcx-v2.13.1-gcc-MLNX_OFED_LINUX-5-ubuntu22.04-cuda11-gdrcopy2-nccl2.12-x86_64
