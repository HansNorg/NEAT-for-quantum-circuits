from time import time
from dataclasses import dataclass
from argparse import ArgumentParser

import numpy as np

from experiments.experimenter import Experimenter, MultipleRunExperimenter
from quantumneat.implementations.linear_growth import LinearGrowthConfig as Config
from quantumneat.implementations.linear_growth import LinearGrowthConfigSeparate as ConfigSeparate
from quantumneat.problems.fox_in_the_hole import new_fitness_n_gates, new_energy, add_encoding_layer
EXPERIMENT_NAME = "linear_growth_fith_fitness"

def no_gradient(self, circuit, n_parameters, parameters, config):
    return 0

@dataclass
class FithConfig(Config):
    fitness_function = new_fitness_n_gates
    encoding_layer = add_encoding_layer
    energy_function = new_energy
    gradient_function = no_gradient

@dataclass
class FithConfigSeparate(ConfigSeparate):
    fitness_function = new_fitness_n_gates
    encoding_layer = add_encoding_layer
    energy_function = new_energy
    gradient_function = no_gradient

def main(n_qubits, population_size, n_generations, folder = "quantumneat", number_of_cpus = -1):
    config = FithConfig(n_qubits, population_size, number_of_cpus = number_of_cpus)
    experimenter = Experimenter(EXPERIMENT_NAME, config, folder=folder)
    experimenter.run_default(n_generations, do_plot=True, do_print=True)
    print(f"Best energy = {experimenter.quantumneat.population.get_best_genome().get_energy()}")

def run_multiple(n_qubits, population_size, n_generations, folder = "quantumneat", number_of_cpus = -1, n_runs = 10):
    config = FithConfig(n_qubits, population_size, number_of_cpus = number_of_cpus)
    experimenter = MultipleRunExperimenter(EXPERIMENT_NAME, config, folder=folder)
    experimenter.run_multiple_experiments(n_runs, n_generations, do_plot_individual=True, do_plot_multiple=True, do_print=True)

def main_separate(n_qubits, population_size, n_generations, folder = "quantumneat", number_of_cpus = -1):
    config = FithConfigSeparate(n_qubits, population_size, number_of_cpus = number_of_cpus)
    experimenter = Experimenter(EXPERIMENT_NAME, config, folder=folder)
    experimenter.run_default(n_generations, do_plot=True, do_print=True)
    print(f"Best energy = {experimenter.quantumneat.population.get_best_genome().get_energy()}")

def run_multiple_separate(n_qubits, population_size, n_generations, folder = "quantumneat", number_of_cpus = -1, n_runs = 10):
    config = FithConfigSeparate(n_qubits, population_size, number_of_cpus = number_of_cpus)
    experimenter = MultipleRunExperimenter(EXPERIMENT_NAME, config, folder=folder)
    experimenter.run_multiple_experiments(n_runs, n_generations, do_plot_individual=True, do_plot_multiple=True, do_print=True)


if __name__ == "__main__":
    np.random.seed(0)
    
    argparser = ArgumentParser()
    argparser.add_argument("-N", "--n_qubits", default=5, type=int, help="Number of qubits")
    argparser.add_argument("-P", "--population_size", default=100, type=int, help="Population size")
    argparser.add_argument("-G", "--generations", default=100, type=int, help="Amount of generations")
    argparser.add_argument("-cpus", "--number_of_cpus", default=-1, type=int, help="Number of cpus to use")
    argparser.add_argument("-R", "--n_runs", default=0, type=int, help="Number of runs (<= 0) means 1 run, but no aggregation of results")
    argparser.add_argument("-rots", "--rots", default="ROT", type=str, help="ROT for one RxRyRz gate, R for separate Rx, Ry and Rz gates.")
    args = argparser.parse_args()

    if args.rots == "ROT":
        if args.n_runs > 0:
            run_multiple(n_qubits=args.n_qubits, population_size=args.population_size, n_generations=args.generations, folder = ".", n_runs=args.n_runs, number_of_cpus=args.number_of_cpus)
        else:
            main(n_qubits=args.n_qubits, population_size=args.population_size, n_generations=args.generations, folder = ".", number_of_cpus=args.number_of_cpus)#"quantumneat"
    elif args.rots == "R":
        if args.n_runs > 0:
            run_multiple_separate(n_qubits=args.n_qubits, population_size=args.population_size, n_generations=args.generations, folder = ".", n_runs=args.n_runs, number_of_cpus=args.number_of_cpus)
        else:
            main_separate(n_qubits=args.n_qubits, population_size=args.population_size, n_generations=args.generations, folder = ".", number_of_cpus=args.number_of_cpus)#"quantumneat"
    else:
        print("Rots arg not recognized", args.rots)