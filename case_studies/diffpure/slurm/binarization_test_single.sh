#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --mem=20G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --output=/home/bethge/rzimmermann/diffpure-slurm-logs/slog-%j.out  # File to which STDOUT will be written
#SBATCH --error=/home/bethge/rzimmermann/diffpure-slurm-logs/slog-%j.err   # File to which STDERR will be written
#SBATCH --gres=gpu:1              # Request right GPUs
#SBATCH --cpus-per-task=8
#SBATCH --job-name=diffpure_binarization_test


function _termmsg()
{
    SLURMDIR="/home/bethge/rzimmermann/diffpure-slurm-logs/"
    echo "terminaged $SLURM_JOB_ID  on $SLURM_JOB_NODELIST. Check slog-$SLURM_JOB_ID.err" > $SLURMDIR/slog-common.err
}

trap _termmsg SIGTERM

arguments=$1

# include information about the job in the output
scontrol show job=$SLURM_JOB_ID

echo "GPU information:"
nvidia-smi --query-gpu=memory.used --format=csv

SCRATCH_DIRECTORY="/scratch_local/$SLURM_JOB_USER-$SLURM_JOBID"

echo "Copying SIF..."
qcopy /mnt/qb/home/bethge/rzimmermann/sifs/diffpure.sif $SCRATCH_DIRECTORY
echo "SIF copied!"

echo "arguments: $1"

log_folder="/home/bethge/rzimmermann/diffpure-logs/"
mkdir -p ${log_folder}

log_path="${log_folder}/${arguments// /_}"

echo "Slurm Job ID: $SLURM_JOB_ID" >> $log_path.out
srun singularity exec --nv --bind /mnt/qb/ --bind $SCRATCH_DIRECTORY "$SCRATCH_DIRECTORY/diffpure.sif" \
  /bin/bash -c "cd /mnt/qb/bethge/rzimmermann/code/active-adversarial-tests-internal && ./case_studies/diffpure/run_scripts/cifar10/run_cifar_rand_inf_binarization_test.sh $arguments 2> ${log_path}.err | tee ${log_path}.log"
