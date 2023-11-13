import logging

import numpy as np
import matplotlib.pyplot as plt

from quantumneat.main import QuantumNEAT
from quantumneat.implementations.qneat import QNEAT_Config
from quantumneat.logger import default_logger

def qneat_experiment(n_generations):
    logger = logging.getLogger("quantumNEAT.experiments.qneat")
    logger.info("qneat_experiment started")
    config = QNEAT_Config(5, 100)
    logger.debug(f"{config.gene_types=}")
    logger.debug(f"{config.compatibility_threshold=}")
    logger.debug(f"{config.dynamic_compatibility_threshold=}")
    quantumneat = QuantumNEAT(config)
    fitness_record, population_size, number_of_species, average_fitnesses = quantumneat.run(n_generations)
    np.savez("./results/results.npz", 
             fitness_record=fitness_record, 
             population_size=population_size, 
             number_of_species=number_of_species, 
             average_fitnesses=average_fitnesses)
    
    plt.plot(fitness_record)
    plt.title("fitness_record")
    # plt.legend()
    plt.savefig("./figures/fitness_record.png")
    plt.close()
    plt.plot(population_size)
    plt.title("population_size")
    # plt.legend()
    plt.savefig("./figures/population_size.png")
    plt.close()
    plt.plot(number_of_species)
    plt.title("number_of_species")
    # plt.legend()
    plt.savefig("./figures/number_of_species.png")
    plt.close()
    plt.plot(average_fitnesses)
    plt.title("average_fitnesses")
    # plt.legend()
    plt.savefig("./figures/average_fitnesses.png")
    plt.close()

if __name__ == "__main__":
    np.random.seed(0)
    default_logger(False)
    qneat_experiment(10)