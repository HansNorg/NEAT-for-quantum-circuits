#!/bin/bash
read -p "Timelimit? " timelimit
read -p "Memorylimit? " memory
read -p "CPUS? " cpus
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$1_$2
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --mail-user="hans.norg99@gmail.com"
#SBATCH --mail-type="ALL"

#SBATCH --partition=cpu_lorentz
#SBATCH --account=cpu_lorentz

#SBATCH --time=$timelimit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$cpus
#SBATCH --mem=$memory

module load ALICE/default
module load Python/3.10.8-GCCcore-12.2.0
source "/home/s3727599/.cache/pypoetry/virtualenvs/quantumneat-gXCYO08V-py3.10/bin/activate"
echo "#### Starting Python test at $(date)"
python ./experiments/run_experiment.py $@
echo "#### Finished Test at $(date)."
EOT