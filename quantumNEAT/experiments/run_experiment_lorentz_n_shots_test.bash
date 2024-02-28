#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$1_$2
#SBATCH --output=slurm_out/%x_%A_%a.out
#SBATCH --mail-user="hans.norg99@gmail.com"
#SBATCH --mail-type="ALL"

#SBATCH --partition=cpu_lorentz
#SBATCH --account=cpu_lorentz
#SBATCH --time=10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

#SBATCH --array=0,1

module load ALICE/default
module load Python/3.10.8-GCCcore-12.2.0
source "/home/s3727599/.cache/pypoetry/virtualenvs/quantumneat-gXCYO08V-py3.10/bin/activate"
echo "[\$SHELL] #### Starting Python test at \$(date)"
python ./experiments/run_experiment.py gs_h2_errorless_saveh linear_growth -N 2 -O 100 --n_shots \$SLURM_ARRAY_TASK_ID -G 2 -P 10
echo "[\$SHELL] #### Finished Test at \$(date)."
EOT