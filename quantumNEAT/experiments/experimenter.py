from __future__ import annotations

import logging
import pickle
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

from quantumneat.logger import default_logger
from quantumneat.main import QuantumNEAT

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from quantumneat.configuration import QuantumNEATConfig

class Experimenter:
    def __init__(self, name:str, config:QuantumNEATConfig, run:int = None, folder:str = "quantumneat") -> None:
        self.config = config
        self.quantumneat = QuantumNEAT(config)
        self.name = name
        self.folder = folder
        if run is None:
            self.load_next_run_number()
        else:
            self.run = run
        default_logger(True, extra_file_name=f"{name}_run{self.run}_")
        self.logger = logging.getLogger(f"quantumNEAT.experiments.{name}")

    def load_next_run_number(self, N = 1):
        try:
            self.run = np.load(f"{self.folder}/experiments/run_cache/{self.name}_run_number.npy")[0]
        except FileNotFoundError:
            self.run = 0
        np.save(f"{self.folder}/experiments/run_cache/{self.name}_run_number", [self.run+N], allow_pickle=False)

    def run_default(self, n_generations, do_plot = False, do_print = True):
        self.logger.info(f"Running experiment {self.name}")
        self.run_generations(n_generations)
        self.save_results()
        if do_plot:
            self.plot_results()
        if do_print:
            self.log_best_circuit()
        self.logger.info(f"Experiment {self.name} finished")

    def run_generations(self, n_generations):
        self.quantumneat.run(n_generations)

    def save_results(self):
        self.logger.info("linear_growth_experiment finished")
        np.savez(f"{self.folder}/results/{self.name}_run{self.run}_results",
                config = self.config,
                fitness_record=self.quantumneat.best_fitnesses, 
                population_size=self.quantumneat.population_sizes, 
                number_of_species=self.quantumneat.number_of_species, 
                average_fitnesses=self.quantumneat.average_fitnesses, 
                best_genomes = self.quantumneat.best_genomes,
                )
        # pickle.dump(best_genomes, folder+"/results/"+name+"best_genomes")
    
    def plot_results(self):
        plt.plot(self.quantumneat.best_fitnesses)
        plt.title("fitness_record")
        plt.savefig(f"{self.folder}/results/{self.name}_run{self.run}_fitness_record.png")
        plt.close()
        plt.plot(self.quantumneat.population_sizes)
        plt.title("population_size")
        plt.savefig(f"{self.folder}/results/{self.name}_run{self.run}_population_size.png")
        plt.close()
        plt.plot(self.quantumneat.number_of_species)
        plt.title("number_of_species")
        plt.savefig(f"{self.folder}/results/{self.name}_run{self.run}_number_of_species.png")
        plt.close()
        plt.plot(self.quantumneat.average_fitnesses)
        plt.title("average_fitnesses")
        plt.savefig(f"{self.folder}/results/{self.name}_run{self.run}_average_fitnesses.png")
        plt.close()
        
    def log_best_circuit(self, fold=-1, do_print = True, do_latex = False):
        if not (do_print or do_latex):
            return
        backup = self.config.simulator
        self.config.simulator = 'qiskit'
        
        for generation, genome in self.quantumneat.best_genomes:
            genome._update_circuit = True
            # genome.config = config
        best_circuits:list[tuple[int, QuantumCircuit]] = [(generation, genome.get_circuit()[0]) for generation, genome in self.quantumneat.best_genomes]

        if do_print:
            best_circuits_drawings = [(generation, circuit.draw(fold=fold)) for generation, circuit in best_circuits]
            for generation, circuit in best_circuits_drawings:
                self.logger.info(f"Generation: {generation}\n{circuit}")
        
        if do_latex:
            best_circuits_latex = [(generation, circuit.draw(output='latex_source', fold=fold)) for generation, circuit in best_circuits]
            for generation, circuit in best_circuits_latex:
                self.logger.info(f"Generation: {generation}\n{circuit}")
        
        self.config.simulator = backup