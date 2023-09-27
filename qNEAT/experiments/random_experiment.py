# Random WIP experiments
import sys
sys.path.append(sys.path[0]+"\\..")
sys.path.append(sys.path[0]+"\\..\qneat")
sys.path.append("/home/s3727599/NEAT-for-quantum-circuits/qNEAT/")
sys.path.append("/home/s3727599/NEAT-for-quantum-circuits/qNEAT/qneat")
from qneat.qNEAT import QNEAT
import qneat.logger as log
import logging
import numpy as np

def main(population_size = 1000, n_runs = 100, n_qubits = 5):
    logger = logging.getLogger("qNEAT.experimenter")
    logger.info("Random experiment started")
    qneat = QNEAT(population_size, n_qubits)
    fitness_record, population_sizes, number_of_species = qneat.run(n_runs)
    np.save("results/fitness_record", fitness_record, allow_pickle=True)
    np.save("results/population_sizes", population_sizes, allow_pickle=True)
    np.save("results/number_of_species", number_of_species, allow_pickle=True)
    logger.info(f"Final circuit: \n{qneat.get_best_circuit().draw(fold=-1)}")
    logger.info("Random experiment finished\n\n\n\n")

if __name__ == "__main__":
    log.QNEATLogger("experiments_random_2", file_level=logging.DEBUG)
    main(500, 20, 5)
    # main()