#!/bin/sh

# bash experiments/run_experiment_lorentz.bash h6_all_errorless_no-solution linear_growth -N 6 -O 200 -R 5
# bash experiments/run_experiment_lorentz.bash lih_all_errorless_no-solution linear_growth -N 10 -O 200 -R 5
# bash experiments/run_experiment_variable.bash gs_h2_errorless linear_growth -N 2 -O 200 -R 5 -G 20
# bash experiments/run_experiment_variable.bash gs_h6_errorless linear_growth -N 6 -O 200 -R 5 -G 20
# bash experiments/run_experiment_variable.bash gs_lih_errorless linear_growth -N 10 -O 200 -G 20
# bash experiments/run_experiment_lorentz_single_cpu.bash gs_h2_errorless_saveh linear_growth -N 2
# bash experiments/run_experiment_lorentz_single_cpu.bash gs_h6_errorless_saveh linear_growth -N 6
# bash experiments/run_experiment_lorentz_single_cpu.bash gs_lih_errorless_saveh linear_growth -N 8

# bash experiments/run_experiment_lorentz_single_cpu.bash gs_h2_errorless_saveh linear_growth -N 2 -O 100
# bash experiments/run_experiment_lorentz_single_cpu.bash gs_h6_errorless_saveh linear_growth -N 6 -O 100
# bash experiments/run_experiment_lorentz_single_cpu.bash gs_lih_errorless_saveh linear_growth -N 8 -O 100

# bash experiments/run_experiment_lorentz_single_cpu.bash gs_h2_errorless_saveh linear_growth -N 2 -O 200
# bash experiments/run_experiment_lorentz_single_cpu.bash gs_h6_errorless_saveh linear_growth -N 6 -O 200
# bash experiments/run_experiment_lorentz_single_cpu.bash gs_lih_errorless_saveh linear_growth -N 8 -O 200

# bash experiments/run_experiment_lorentz_single_cpu.bash gs_h2_errorless_saveh qneat -N 2
# bash experiments/run_experiment_lorentz_single_cpu.bash gs_h6_errorless_saveh qneat -N 6
# bash experiments/run_experiment_lorentz_single_cpu.bash gs_lih_errorless_saveh qneat -N 8

# bash experiments/run_experiment_lorentz_single_cpu.bash gs_h2_errorless_saveh qneat -N 2 -O 100
# bash experiments/run_experiment_lorentz_single_cpu.bash gs_h6_errorless_saveh qneat -N 6 -O 100
# bash experiments/run_experiment_lorentz_single_cpu.bash gs_lih_errorless_saveh qneat -N 8 -O 100

# bash experiments/run_experiment_lorentz_single_cpu.bash gs_h2_errorless_saveh qneat -N 2 -O 200
# bash experiments/run_experiment_lorentz_single_cpu.bash gs_h6_errorless_saveh qneat -N 6 -O 200
# bash experiments/run_experiment_lorentz_single_cpu.bash gs_lih_errorless_saveh qneat -N 8 -O 200

# bash experiments/run_experiment_lorentz_n_shots.bash gs_h2_errorless_saveh linear_growth -N 2 -O 100
# bash experiments/run_experiment_lorentz_n_shots.bash gs_h6_errorless_saveh linear_growth -N 6 -O 100
# bash experiments/run_experiment_lorentz_n_shots.bash gs_lih_errorless_saveh linear_growth -N 8 -O 100

# bash experiments/run_experiment_lorentz_n_shots.bash gs_h2_errorless_saveh qneat -N 2 -O 100
# bash experiments/run_experiment_lorentz_n_shots.bash gs_h6_errorless_saveh qneat -N 6 -O 100
# bash experiments/run_experiment_lorentz_n_shots.bash gs_lih_errorless_saveh qneat -N 8 -O 100

# bash experiments/run_experiment_lorentz_n_shots.bash gs_h2_errorless_saveh linear_growth -N 2 -O 100 --phys_noise
# bash experiments/run_experiment_lorentz_n_shots.bash gs_h6_errorless_saveh linear_growth -N 6 -O 100 --phys_noise
# bash experiments/run_experiment_lorentz_n_shots.bash gs_lih_errorless_saveh linear_growth -N 8 -O 100 --phys_noise

# bash experiments/run_experiment_lorentz_n_shots.bash gs_h2_errorless_saveh qneat -N 2 -O 100 --phys_noise
# bash experiments/run_experiment_lorentz_n_shots.bash gs_h6_errorless_saveh qneat -N 6 -O 100 --phys_noise
# bash experiments/run_experiment_lorentz_n_shots.bash gs_lih_errorless_saveh qneat -N 8 -O 100 --phys_noise

# bash experiments/run_experiment_n_shots.bash gs_h2_errorless_saveh linear_growth -N 2 -O 100 -G 20 -R 11 
# bash experiments/run_experiment_n_shots.bash gs_h6_errorless_saveh linear_growth -N 6 -O 100 -G 20 -R 11

# bash experiments/run_experiment_n_shots.bash gs_h2_errorless_saveh linear_growth -N 2 -O 100 -G 20 -R 11 --phys_noise
# bash experiments/run_experiment_n_shots.bash gs_h6_errorless_saveh linear_growth -N 6 -O 100 -G 20 -R 11 --phys_noise

bash experiments/run_experiment_n_shots.bash gs_h2_errorless_saveh linear_growth -N 2 -O 100 -G 20 -R 11 