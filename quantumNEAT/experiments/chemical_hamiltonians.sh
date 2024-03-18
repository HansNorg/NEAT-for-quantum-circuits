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

# bash experiments/run_experiment_n_shots.bash gs_h2_errorless_saveh linear_growth -N 2 -O 100 -G 20 -R 11 

bash experiments/run_experiment_single_cpu.bash gs_h2_errorless_saveh linear_growth -N 2 -O 100 -G 20 -R 10 --phys_noise # Finished 1100/experiment
bash experiments/run_experiment_single_cpu.bash gs_h6_errorless_saveh linear_growth -N 6 -O 100 -G 20 -R 10 --phys_noise # Mixed -> TimeOut Run 3 Gen 18
bash experiments/run_experiment_single_cpu.bash gs_lih_errorless_saveh linear_growth -N 8 -O 100 -G 20 --phys_noise # Failed -> TimeOut Gen 10
bash experiments/run_experiment_single_cpu.bash gs_lih_errorless_saveh linear_growth -N 8 -O 100 -G 20 --phys_noise # Failed -> TimeOut Gen 10

bash experiments/run_experiment_single_cpu.bash gs_h2_errorless_saveh linear_growth -N 2 -O 100 -G 20 -R 10 --phys_noise --total_energy # Finished 1600/experiment
bash experiments/run_experiment_single_cpu.bash gs_h6_errorless_saveh linear_growth -N 6 -O 100 -G 20 -R 10 --phys_noise --total_energy # Mixed -> TimeOut Run 3 Gen 13
bash experiments/run_experiment_single_cpu.bash gs_lih_errorless_saveh linear_growth -N 8 -O 100 -G 20 --phys_noise --total_energy # Failed -> TimeOut Gen 10

bash experiments/run_experiment_single_cpu.bash gs_h2_errorless_saveh linear_growth -N 2 -O 100 -G 100 -R 10 --total_energy # Finished 5900/experiment
bash experiments/run_experiment_single_cpu.bash gs_h6_errorless_saveh linear_growth -N 6 -O 100 -G 100 -R 10 --total_energy # Mixed -> TimeOut Run 9 Gen 25
bash experiments/run_experiment_single_cpu.bash gs_lih_errorless_saveh linear_growth -N 8 -O 100 -G 100 -R 10 --total_energy # Mixed -> TimeOut Run 8 Gen 28

bash experiments/run_experiment_single_cpu.bash gs_h2_errorless_saveh linear_growth -N 2 -O 100 -G 100 -R 10 # Finished 3500/experiment
bash experiments/run_experiment_single_cpu.bash gs_h6_errorless_saveh linear_growth -N 6 -O 100 -G 100 -R 10 # Finished 7400/experiment
bash experiments/run_experiment_single_cpu.bash gs_lih_errorless_saveh linear_growth -N 8 -O 100 -G 100 -R 10 # Mixed -> TimeOut Run 9 Gen 44


bash experiments/run_experiment_single_cpu.bash gs_h2_errorless_saveh linear_growth -N 2 -O 100 -G 20 -R 10 --phys_noise --fitness_sharing # Finished 1100/experiment
bash experiments/run_experiment_single_cpu.bash gs_h6_errorless_saveh linear_growth -N 6 -O 100 -G 20 -R 10 --phys_noise --fitness_sharing # Mixed -> TimeOut Run 3 Gen 19
bash experiments/run_experiment_single_cpu.bash gs_lih_errorless_saveh linear_growth -N 8 -O 100 -G 20 --phys_noise --fitness_sharing # Failed -> TimeOut Gen 9
bash experiments/run_experiment_single_cpu.bash gs_lih_errorless_saveh linear_growth -N 8 -O 100 -G 20 --phys_noise --fitness_sharing # Failed -> TimeOut Gen 9

bash experiments/run_experiment_single_cpu.bash gs_h2_errorless_saveh linear_growth -N 2 -O 100 -G 20 -R 10 --phys_noise --total_energy --fitness_sharing # Mixed -> ValueError Run 5 Gen 16
bash experiments/run_experiment_single_cpu.bash gs_h6_errorless_saveh linear_growth -N 6 -O 100 -G 20 -R 10 --phys_noise --total_energy --fitness_sharing # Mixed -> TimeOut Run 3 Gen 14
bash experiments/run_experiment_single_cpu.bash gs_lih_errorless_saveh linear_growth -N 8 -O 100 -G 20 --phys_noise --total_energy --fitness_sharing # Failed -> TimeOut Gen 9

bash experiments/run_experiment_single_cpu.bash gs_h2_errorless_saveh linear_growth -N 2 -O 100 -G 100 -R 10 --total_energy --fitness_sharing # Failed -> ValueError Gen 60
bash experiments/run_experiment_single_cpu.bash gs_h6_errorless_saveh linear_growth -N 6 -O 100 -G 100 -R 10 --total_energy --fitness_sharing # Failed -> ValueError Gen 57
bash experiments/run_experiment_single_cpu.bash gs_lih_errorless_saveh linear_growth -N 8 -O 100 -G 100 -R 10 --total_energy --fitness_sharing # Failed -> ValueError Gen 67

bash experiments/run_experiment_single_cpu.bash gs_h2_errorless_saveh linear_growth -N 2 -O 100 -G 100 -R 10 --fitness_sharing # Failed -> ValueError Gen 62
bash experiments/run_experiment_single_cpu.bash gs_h6_errorless_saveh linear_growth -N 6 -O 100 -G 100 -R 10 --fitness_sharing # Mixed -> ValueError Run 2 Gen 40
bash experiments/run_experiment_single_cpu.bash gs_lih_errorless_saveh linear_growth -N 8 -O 100 -G 100 -R 10 --fitness_sharing # Mixed -> ValueError Run 2 Gen 38