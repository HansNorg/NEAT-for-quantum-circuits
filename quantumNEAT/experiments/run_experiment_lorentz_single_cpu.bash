#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$1_$2
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --mail-user="hans.norg99@gmail.com"
#SBATCH --mail-type="ALL"

#SBATCH --partition=cpu_lorentz
#SBATCH --account=cpu_lorentz
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

module load ALICE/default
module load Python/3.10.8-GCCcore-12.2.0
source "/home/s3727599/.cache/pypoetry/virtualenvs/quantumneat-gXCYO08V-py3.10/bin/activate"
echo "[\$SHELL] #### Starting Python test at \$(date)"
echo $@
python ./experiments/run_experiment.py $@
echo "[\$SHELL] #### Finished Test at \$(date)."
EOT