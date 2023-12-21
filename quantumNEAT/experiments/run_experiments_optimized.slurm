#!/bin/bash
#SBATCH --job-name=TFI_quantumNEAT_linear_optimized
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --mail-user="hans.norg99@gmail.com"
#SBATCH --mail-type="ALL"

#SBATCH --partition="cpu-medium"
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=300M

module load Python/3.10.8-GCCcore-12.2.0
source "/home/s3727599/.cache/pypoetry/virtualenvs/quantumneat-gXCYO08V-py3.10/bin/activate"
DATE=$(date)
NAME=$(%x)
echo "#### Starting $NAME test at $DATE"
python ./experiments/linear_growth_optimized.py
DATE=$(date)
echo "#### Finished Test at $DATE."