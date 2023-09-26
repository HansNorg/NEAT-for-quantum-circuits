# Random WIP experiments
import sys
sys.path.append(sys.path[0]+"\\..")
sys.path.append(sys.path[0]+"\\..\qneat")
from qneat.qNEAT import QNEAT
import qneat.logger as log
import logging

def main(population_size = 1000, n_runs = 1000, n_qubits = 5):
    logger = logging.getLogger("qNEAT.experimenter")
    logger.info("Random experiment started")
    qneat = QNEAT(population_size, n_qubits)
    qneat.run(n_runs)
    logger.info(f"Final circuit: \n{qneat.get_best_circuit()}")
    logger.info("Random experiment finished\n\n\n\n")

if __name__ == "__main__":
    log.QNEATLogger("experiments_random", file_level=logging.DEBUG)
    main(1000, 10, 5)
    # main()