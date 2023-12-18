#!/bin/bash
#SBATCH --job-name=linear_fith_separate
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --mail-user="hans.norg99@gmail.com"
#SBATCH --mail-type="ALL"

#SBATCH --partition="cpu-medium"
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

module load Python/3.10.8-GCCcore-12.2.0
source "/home/s3727599/.cache/pypoetry/virtualenvs/quantumneat-gXCYO08V-py3.10/bin/activate"
DATE=$(date)
echo "#### Starting Python test at $DATE"
python ./experiments/linear_growth_fith_new.py -G 1000 -R 10 -cpus 2 -rots "R"
DATE=$(date)
echo "#### Finished Test at $DATE."