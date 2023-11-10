import logging

import numpy as np
import matplotlib.pyplot as plt

from quantumneat.main import QuantumNEAT
from quantumneat.implementations.qneat import QNEAT_Config
from quantumneat.logger import default_logger

def qneat_experiment():
    logger = logging.getLogger("quantumNEAT.experiments.qneat")
    logger.info("qneat_experiment started")
    config = QNEAT_Config(5, 100)
    logger.debug(config.gene_types)
    logger.debug(config.compatibility_threshold)
    logger.debug(config.dynamic_compatibility_threshold)
    quantumneat = QuantumNEAT(config)
    fitness_record, population_size, number_of_species, average_fitnesses = quantumneat.run(10)
    np.savez("./results", 
             fitness_record=fitness_record, 
             population_size=population_size, 
             number_of_species=number_of_species, 
             average_fitnesses=average_fitnesses)
    
    plt.plot(fitness_record)
    plt.title("fitness_record")
    plt.show()
    plt.plot(population_size)
    plt.title("population_size")
    plt.show()
    plt.plot(number_of_species)
    plt.title("number_of_species")
    plt.show()
    plt.plot(average_fitnesses)
    plt.title("average_fitnesses")
    plt.show()

if __name__ == "__main__":
    default_logger(True)
    qneat_experiment()