#!/bin/sh
python experiments/run_experiment.py tfim linear_growth -O 100 -R 10&
python experiments/run_experiment.py tfim linear_growth -O 100 -R 10 -X "no-force"&
python experiments/run_experiment.py tfim linear_growth -O 100 -R 10 -X "normalise"&
python experiments/run_experiment.py tfim linear_growth -O 100 -R 10 -X "no-force, normalise"&
wait
echo finished