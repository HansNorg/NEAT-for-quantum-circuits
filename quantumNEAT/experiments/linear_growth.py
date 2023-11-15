import logging
import sys

import numpy as np
import matplotlib.pyplot as plt

from quantumneat.main import QuantumNEAT
from quantumneat.implementations.linear_growth import LinearGrowthConfig
from quantumneat.logger import default_logger, setup_logger

def qneat_experiment(n_generations, name, folder = "quantumneat"):
    logger = logging.getLogger("quantumNEAT.experiments.linear_growth")
    logger.info("linear_growth_experiment started")
    config = LinearGrowthConfig(5, 100)
    quantumneat = QuantumNEAT(config)
    fitness_record, population_size, number_of_species, average_fitnesses = quantumneat.run(n_generations)
    np.savez(folder+"/results/"+name+"results",
             fitness_record=fitness_record, 
             population_size=population_size, 
             number_of_species=number_of_species, 
             average_fitnesses=average_fitnesses)
    np.save(folder+"/results/"+name+"config", config)
    
    plt.plot(fitness_record)
    plt.title("fitness_record")
    # plt.legend()
    plt.savefig(folder+"/figures/"+name+"fitness_record.png")
    plt.close()
    plt.plot(population_size)
    plt.title("population_size")
    # plt.legend()
    plt.savefig(folder+"/figures/"+name+"population_size.png")
    plt.close()
    plt.plot(number_of_species)
    plt.title("number_of_species")
    # plt.legend()
    plt.savefig(folder+"/figures/"+name+"number_of_species.png")
    plt.close()
    plt.plot(average_fitnesses)
    plt.title("average_fitnesses")
    # plt.legend()
    plt.savefig(folder+"/figures/"+name+"average_fitnesses.png")
    plt.close()
    logger.info("linear_growth_experiment finished")

if __name__ == "__main__":
    np.random.seed(0)
    folder = "."#"quantumneat"
    try:
        run_n = np.load(folder+"/experiments/linear_growth_run_number.npy")[0]+1
    except FileNotFoundError:
        run_n = 0
    np.save(folder+"/experiments/linear_growth_run_number", [run_n])
    name = f"linear_growth_run{run_n}_"
    default_logger(True, extra_file_name=name)
    # setup_logger(quantumneat_level=logging.DEBUG)
    qneat_experiment(100, name, folder)