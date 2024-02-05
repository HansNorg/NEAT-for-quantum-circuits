#!/bin/sh

for i in $(seq 0.2 0.05 2.85); do
    bash experiments/run_experiment_lorentz.bash h2_R_$i linear_growth -N 2 -O 100
done