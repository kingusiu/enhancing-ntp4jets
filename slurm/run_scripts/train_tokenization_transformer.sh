#!/bin/sh

#SBATCH --job-name=train-tokenization-transformer
#SBATCH --time=38:00:00
#SBATCH --partition=private-dpnc-gpu
#SBATCH --mem=50GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus=1
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
  /home/users/w/wozniak/container/omni_alfa_continuous.sif \
  bash -c "source /opt/conda/bin/activate && export PYTHONPATH=/home/users/w/wozniak/.local/ntp4jets-deps:\$PYTHONPATH && python gabbro/train.py experiment=example_experiment_tokenization_transformer"
