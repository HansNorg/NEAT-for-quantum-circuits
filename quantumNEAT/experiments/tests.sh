#!/bin/sh
echo ======= Started ======== \n\n
python experiments/run_experiment.py tfim linear_growth -O 100 -R 10&
python experiments/run_experiment.py tfim linear_growth -O 100 -R 10 -X "no-force"&
python experiments/run_experiment.py tfim linear_growth -O 100 -R 10 -X "normalise"&
python experiments/run_experiment.py tfim linear_growth -O 100 -R 10 -X "no-force_normalise"&
wait
echo \n\n ======= Finished ========