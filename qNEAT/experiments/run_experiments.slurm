#!/bin/bash
#SBATCH --job-name=1st_experiments_qneat
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --mail-user="hans.norg99@gmail.com"
#SBATCH --mail-type="ALL"

#SBATCH --partition="cpu-short"
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --mem=200M

module load Python/3.10.8-GCCcore-12.2.0
source "/home/s3727599/.cache/pypoetry/virtualenvs/qneat-6v_DPKM0-py3.10/bin/activate"
DATE=$(date)
echo "#### Starting Python test at $DATE"
python ./experiments/random_experiment.py
DATE=$(date)
echo "#### Finished Test at $DATE."