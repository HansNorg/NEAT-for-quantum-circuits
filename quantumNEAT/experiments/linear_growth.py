import numpy as np

from experiments.experimenter import Experimenter
from quantumneat.implementations.linear_growth import LinearGrowthConfig as Config

EXPERIMENT_NAME = "linear_growth"

def main(n_qubits, population_size, n_generations, folder = "quantumneat"):
    config = Config(n_qubits, population_size)
    experimenter = Experimenter(EXPERIMENT_NAME, config, folder=folder)
    experimenter.run_default(n_generations, do_plot=True, do_print=True)

if __name__ == "__main__":
    np.random.seed(0)
    main(n_qubits=5, population_size=100, n_generations=1000, folder = ".")#"quantumneat"