#!/bin/bash

for i in $(seq 0.2 0.05 2.85); do
    python experiments/run_experiment.py h2_R_$i linear_growth -N 2 -O 100 -G 10
done