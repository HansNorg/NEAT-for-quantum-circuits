import numpy as np

from experiments.experimenter import Experimenter, MultipleRunExperimenter
from quantumneat.implementations.linear_growth import LinearGrowthConfig as Config
from qulacs import ParametricQuantumCircuit
from quantumneat.helper import get_energy_qulacs
EXPERIMENT_NAME = "linear_growth"

def main(n_qubits, population_size, n_generations, folder = "quantumneat"):
    config = Config(n_qubits, population_size)
    experimenter = Experimenter(EXPERIMENT_NAME, config, folder=folder)
    experimenter.run_default(n_generations, do_plot=True, do_print=True)
    print(experimenter.quantumneat.population.get_best_genome().get_energy())

def run_multiple(n_qubits, population_size, n_generations, folder = "quantumneat", n_runs = 10):
    config = Config(n_qubits, population_size)
    experimenter = MultipleRunExperimenter(EXPERIMENT_NAME, config, folder=folder)
    experimenter.run_multiple_experiments(n_runs, n_generations, do_plot_individual=True, do_plot_multiple=True, do_print=True)

if __name__ == "__main__":
    np.random.seed(0)
    # main(n_qubits=5, population_size=100, n_generations=200, folder = ".")#"quantumneat"
    run_multiple(n_qubits=5, population_size=100, n_generations=100, folder = ".", n_runs=10)