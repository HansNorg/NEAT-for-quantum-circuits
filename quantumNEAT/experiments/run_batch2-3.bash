#!/bin/bash
#SBATCH --job-name=h2_batch
#SBATCH --output=slurm_out/%x_%A_%a.out
#SBATCH --mail-user="hans.norg99@gmail.com"
#SBATCH --mail-type="ALL"

#SBATCH --partition=cpu-medium
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

#SBATCH --array=0-27

module load ALICE/default
module load Python/3.10.8-GCCcore-12.2.0
source "/home/s3727599/.cache/pypoetry/virtualenvs/quantumneat-gXCYO08V-py3.10/bin/activate"
echo "[$SHELL] #### Starting Python test at $(date)"
python ./experiments/run_batch.py 2 3 $SLURM_ARRAY_TASK_ID
echo "[$SHELL] #### Finished Test at $(date)."