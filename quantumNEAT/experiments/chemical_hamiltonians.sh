#!/bin/sh

# bash experiments/run_experiment_lorentz.bash h6_all_errorless_no-solution linear_growth -N 6 -O 200 -R 5
# bash experiments/run_experiment_lorentz.bash lih_all_errorless_no-solution linear_growth -N 10 -O 200 -R 5
bash experiments/run_experiment_variable.bash gs_h2_errorless linear_growth -N 2 -O 200 -R 5
bash experiments/run_experiment_variable.bash gs_h6_errorless linear_growth -N 6 -O 200 -R 5
bash experiments/run_experiment_variable.bash gs_lih_errorless linear_growth -N 10 -O 200