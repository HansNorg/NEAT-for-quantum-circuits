#!/bin/sh
py-spy record -o testpyspy.svg --subprocesses -- python experiments/run_experiment.py tfim linear_growth -X "speed_"&
py-spy record -o testpyspy_no-force.svg --subprocesses -- python experiments/run_experiment.py tfim linear_growth -X "speed_no-force"&
py-spy record -o testpyspy_normalise.svg --subprocesses -- python experiments/run_experiment.py tfim linear_growth -X "speed_normalise"&
py-spy record -o testpyspy-no-force_normalise.svg --subprocesses -- python experiments/run_experiment.py tfim linear_growth -X "speed_no-force_normalise"&
wait
echo finished