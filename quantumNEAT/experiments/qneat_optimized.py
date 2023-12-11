from dataclasses import dataclass

import numpy as np

from experiments.experimenter import Experimenter, MultipleRunExperimenter
from quantumneat.implementations.qneat import QNEAT_Config as Config

EXPERIMENT_NAME = "qneat_optimized"

@dataclass
class OptConfig(Config):
    optimize_energy = True

def main(n_qubits, population_size, n_generations, folder = "quantumneat"):
    config = OptConfig(n_qubits, population_size)
    experimenter = Experimenter(EXPERIMENT_NAME, config, folder=folder)
    experimenter.run_default(n_generations, do_plot=True, do_print=True)
    print(experimenter.quantumneat.population.get_best_genome().get_energy())

def run_multiple(n_qubits, population_size, n_generations, folder = "quantumneat", n_runs = 10):
    config = OptConfig(n_qubits, population_size)
    experimenter = MultipleRunExperimenter(EXPERIMENT_NAME, config, folder=folder)
    experimenter.run_multiple_experiments(n_runs, n_generations, do_plot_individual=True, do_plot_multiple=True, do_print=True)

def run_multiple_empty(n_qubits, population_size, n_generations, folder = "quantumneat", n_runs = 10):
    config = OptConfig(n_qubits, population_size, initial_layers=0)
    experimenter = MultipleRunExperimenter(EXPERIMENT_NAME+"_empty", config, folder=folder)
    experimenter.run_multiple_experiments(n_runs, n_generations, do_plot_individual=True, do_plot_multiple=True, do_print=True)

if __name__ == "__main__":
    np.random.seed(0)
    # main(n_qubits=5, population_size=100, n_generations=200, folder = ".")#"quantumneat"
    # run_multiple(n_qubits=5, population_size=100, n_generations=1000, folder = ".", n_runs=10)
    run_multiple_empty(n_qubits=5, population_size=100, n_generations=100, folder = ".", n_runs=10)