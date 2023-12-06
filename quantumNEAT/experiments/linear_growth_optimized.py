from dataclasses import dataclass
from argparse import ArgumentParser
from time import time

import numpy as np

from experiments.experimenter import Experimenter, MultipleRunExperimenter
from quantumneat.implementations.linear_growth import LinearGrowthConfig as Config
from qulacs import ParametricQuantumCircuit
from quantumneat.helper import get_energy_qulacs
# EXPERIMENT_NAME = "linear_growth_optimized"
EXPERIMENT_NAME = "linear_growth_optimized_energy_difference_positive"

@dataclass
class OptConfig(Config):
    optimize_energy = True

def main(n_qubits, population_size, n_generations, folder = "quantumneat", number_of_cpus = -1):
    config = OptConfig(n_qubits, population_size, number_of_cpus=number_of_cpus)
    experimenter = Experimenter(EXPERIMENT_NAME, config, folder=folder)
    experimenter.run_default(n_generations, do_plot=True, do_print=True)
    print(experimenter.quantumneat.population.get_best_genome().get_energy())

def run_multiple(n_qubits, population_size, n_generations, folder = "quantumneat", n_runs = 10):
    config = OptConfig(n_qubits, population_size)
    experimenter = MultipleRunExperimenter(EXPERIMENT_NAME, config, folder=folder)
    experimenter.run_multiple_experiments(n_runs, n_generations, do_plot_individual=True, do_plot_multiple=True, do_print=True)

def test_multiprocessing(population_size, n_generations, number_of_cpus=-1):
    starttime = time()
    main(n_qubits=5, population_size=population_size, n_generations=n_generations, folder = ".", number_of_cpus = number_of_cpus)#"quantumneat"
    print(f"5 qubits, {population_size} population, {n_generations} generations, {number_of_cpus} cpus, time = {time()-starttime}")

if __name__ == "__main__":
    np.random.seed(0)
    CPUS = 4
    
    argparser = ArgumentParser()
    argparser.add_argument("-P", "--population_size", default=100, type=int, help="Population size")
    argparser.add_argument("-G", "--generations", default=100, type=int, help="Amount of generations")
    argparser.add_argument("-cpus", "--number_of_cpus", default=-1, type=int, help="Number of cpus to use")
    args = argparser.parse_args()

    test_multiprocessing(args.population_size, args.generations, args.number_of_cpus)

# if __name__ == "__main__":
#     np.random.seed(0)
#     # main(n_qubits=10, population_size=100, n_generations=2, folder = ".")#"quantumneat"
#     run_multiple(n_qubits=5, population_size=100, n_generations=100, folder = ".", n_runs=10)