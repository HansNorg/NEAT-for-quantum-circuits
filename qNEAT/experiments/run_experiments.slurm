#!/bin/bash
#SBATCH --job-name=test_helloworld
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="hans.norg99@gmail.com"
#SBATCH --mail-type="ALL"

#SBATCH --partition="testing"
#SBATCH --time=00:00:15
#SBATCH --ntasks=1
#SBATCH --mem=10M

echo "#### Starting Test"
echo "This is $SLURM_JOB_USER and my first job has the ID $SLURM_JOB_ID"
# get the current working directory
CWD=$(pwd)
echo "This job was submitted from $SLURM_SUBMIT_DIR and I am currently in $CWD"
# get the current time and date
DATE=$(date)
echo "It is now $DATE"
echo "Hello World from $HOSTNAME"
echo "#### Finished Test. Have a nice day"