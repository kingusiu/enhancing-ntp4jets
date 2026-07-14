#!/bin/sh

#SBATCH --job-name=train-tokenization-transformer
#SBATCH --time=72:00:00
#SBATCH --partition=private-dpnc-gpu
#SBATCH --mem=40GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=4
#SBATCH --output=/home/users/w/wozniak/dev/enhancing-ntp4jets/slurm/logs/slurm-%A-%x.out
#SBATCH --chdir=/home/users/w/wozniak/dev/enhancing-ntp4jets


echo 'Your job is running on node(s):'
echo $SLURM_JOB_NODELIST
echo 'Tasks per node:'
echo $SLURM_TASKS_PER_NODE

echo "Allocated session on node '`hostname -s`'."

export XDG_RUNTIME_DIR=""
export PYTHONPATH=${PWD}:${PWD}/python_install:${PYTHONPATH}

# NOTE: sourcing conda and setting PYTHONPATH does not persist automatically
# inside the container (unlike an interactive `apptainer shell` session where
# these need to be run manually), so they are explicitly chained into the
# same `bash -c` invocation below.
srun apptainer exec \
  --nv -B /srv,/home \
  --env PYTHONNOUSERSITE=1 \
  --env TORCH_NCCL_TRACE_BUFFER_SIZE=2000 \
  --env TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
  --env TORCH_DISTRIBUTED_DEBUG=DETAIL \
  /home/users/w/wozniak/container/omni_alfa_continuous.sif \
  bash -c "source /opt/conda/bin/activate && export PYTHONPATH=/home/users/w/wozniak/.local/ntp4jets-deps:\$PYTHONPATH && python gabbro/train.py experiment=example_experiment_tokenization_transformer trainer=ddp"
