#!/bin/bash
#SBATCH --job-name=1st_experiments_qneat
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --mail-user="hans.norg99@gmail.com"
#SBATCH --mail-type="ALL"

#SBATCH --partition="testing"
#SBATCH --time=00:00:10
#SBATCH --ntasks=1
#SBATCH --mem=100M

module load Python/3.10.8-GCCcore-12.2.0
poetry shell


echo "#### Starting Python test"
python experiments/random_experiment.py
echo "#### Finished Test."