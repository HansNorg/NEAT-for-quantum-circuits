#!/bin/bash
#SBATCH --job-name=linear_optimized_TFI
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --mail-user="hans.norg99@gmail.com"
#SBATCH --mail-type="ALL"

#SBATCH --partition="cpu-medium"
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=500M

module load Python/3.10.8-GCCcore-12.2.0
source "/home/s3727599/.cache/pypoetry/virtualenvs/quantumneat-gXCYO08V-py3.10/bin/activate"
DATE=$(date)
NAME=$(%x)
echo "#### Starting $NAME test at $DATE"
python ./experiments/linear_growth_optimized.py -G 1000 -P 100 -R 10
# python ./experiments/linear_growth_optimized.py -G 100 -P 1000 -R 10
DATE=$(date)
echo "#### Finished Test at $DATE."