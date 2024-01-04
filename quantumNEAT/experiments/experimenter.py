from __future__ import annotations

import os
import logging
import pickle
from time import time
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from quantumneat.logger import default_logger
from quantumneat.main import QuantumNEAT

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from quantumneat.configuration import QuantumNEATConfig

class Experimenter:
    def __init__(self, name:str, config:QuantumNEATConfig, run:int = None, folder:str = "quantumneat", setup_logger = True) -> None:
        self.config = config
        self.name = name
        self.folder = folder
        if run is None:
            self.load_next_run_number()
        else:
            self.run = run
        print(f"Run {self.run} initialised")
        if setup_logger:
            default_logger(True, extra_file_name=f"{name}_run{self.run}_")
        self.logger = logging.getLogger(f"quantumNEAT.experiments.{name}_run{self.run}")
        
        self.quantumneat = QuantumNEAT(config)
        self.final_energies = []
    
    def load_next_run_number(self, N = 1):
        try:
            self.run = np.load(f"{self.folder}/experiments/run_cache/{self.name}_run_number.npy")[0]
        except FileNotFoundError:
            self.run = 0
        os.makedirs(f"{self.folder}/experiments/run_cache", exist_ok=True)
        np.save(f"{self.folder}/experiments/run_cache/{self.name}_run_number", [self.run+N], allow_pickle=False)
    
    def reset_run_number(self, new_number = 0):
        os.makedirs(f"{self.folder}/experiments/run_cache", exist_ok=True)
        np.save(f"{self.folder}/experiments/run_cache/{self.name}_run_number", [new_number], allow_pickle=False)

    def run_default(self, n_generations, do_plot = False, do_print = True):
        self.logger.info(f"Running experiment {self.name}")
        starttime = time()
        print(f"running experiment {self.name}")
        # self.run_generations(n_generations)
        # self.final_energies = self.quantumneat.get_energies()
        # self.save_results()
        # if do_plot:
        #     self.plot_results()
        # if do_print:
        #     self.log_best_circuit()
        runtime = time() - starttime
        self.logger.info(f"Experiment {self.name} finished in {runtime}.")

    def run_generations(self, n_generations):
        self.quantumneat.run(n_generations)

    def save_results(self):
        self.logger.info("linear_growth_experiment finished")
        os.makedirs(f"{self.folder}/results", exist_ok=True)
        np.savez(f"{self.folder}/results/{self.name}_run{self.run}_results",
                config = self.config,
                fitness_record=self.quantumneat.best_fitnesses, 
                population_size=self.quantumneat.population_sizes, 
                number_of_species=self.quantumneat.number_of_species, 
                average_fitnesses=self.quantumneat.average_fitnesses, 
                best_genomes = self.quantumneat.best_genomes,
                best_energies = self.quantumneat.best_energies,
                final_energies = self.final_energies,
                number_of_solutions = self.quantumneat.number_of_solutions,
                min_energies = self.quantumneat.min_energies,
                )
        # pickle.dump(best_genomes, folder+"/results/"+name+"best_genomes")
    
    def plot_results(self):
        os.makedirs(f"{self.folder}/figures", exist_ok=True)
        plt.plot(self.quantumneat.best_fitnesses)
        plt.title("fitness_record")
        plt.xlabel("Generations")
        plt.savefig(f"{self.folder}/figures/{self.name}_run{self.run}_fitness_record.png")
        plt.close()
        plt.plot(self.quantumneat.population_sizes)
        plt.title("population_size")
        plt.xlabel("Generations")
        plt.savefig(f"{self.folder}/figures/{self.name}_run{self.run}_population_size.png")
        plt.close()
        plt.plot(self.quantumneat.number_of_species)
        plt.title("number_of_species")
        plt.xlabel("Generations")
        plt.savefig(f"{self.folder}/figures/{self.name}_run{self.run}_number_of_species.png")
        plt.close()
        plt.plot(self.quantumneat.average_fitnesses)
        plt.title("average_fitnesses")
        plt.xlabel("Generations")
        plt.savefig(f"{self.folder}/figures/{self.name}_run{self.run}_average_fitnesses.png")
        plt.close()
        plt.plot(self.quantumneat.best_energies)
        plt.title("best_energies")
        plt.xlabel("Generations")
        plt.savefig(f"{self.folder}/figures/{self.name}_run{self.run}_best_energies.png")
        plt.close()
        plt.hist(self.final_energies)
        # plt.hist(self.final_energies, bins=np.arange(np.floor(min(self.final_energies)), np.ceil(max(self.final_energies)))-0.5)
        plt.title("final_energies")
        plt.xlabel("Energy")
        plt.ylabel("Number of circuits in final population")
        plt.savefig(f"{self.folder}/figures/{self.name}_run{self.run}_final_energies.png")
        plt.close()
        plt.plot(self.quantumneat.number_of_solutions)
        plt.title("number_of_solutions")
        plt.xlabel("Generations")
        plt.savefig(f"{self.folder}/figures/{self.name}_run{self.run}_number_of_solutions.png")
        plt.close()
        plt.plot(self.quantumneat.min_energies)
        plt.title("min_energies")
        plt.xlabel("Generations")
        plt.savefig(f"{self.folder}/figures/{self.name}_run{self.run}_min_energies.png")
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

class MultipleRunExperimenter:
    def __init__(self, name:str, config:QuantumNEATConfig, folder:str = "quantumneat") -> None:
        self.config = config
        self.name = name
        self.folder = folder
        self.experimenters:list[Experimenter] = []

        default_logger(True, extra_file_name=f"{name}_multiple_runs_")
        self.logger = logging.getLogger(f"quantumNEAT.experiments.{name}_multiple_runs")

    def run_experiment(self, n_generations, do_plot = False, do_print = True):
        experimenter = Experimenter(self.name, self.config, folder=self.folder, setup_logger=False)
        experimenter.run_default(n_generations, do_plot, do_print)
        self.experimenters.append(experimenter)

    def run_multiple_experiments(self, n_experiments, n_generations, do_plot_individual=False, do_plot_multiple = True, do_print=False):
        starttime = time()
        for i in range(n_experiments):
            self.logger.info(f"Running experiment {i+1}/{n_experiments}")
            self.run_experiment(n_generations, do_plot_individual, do_print)
        runtime = time() - starttime
        self.logger.info(f"Finished running {n_experiments} experiments in {runtime} time. ({runtime/n_experiments} time/experiment)")
        if do_plot_multiple:
            try:
                self.plot_multiple()
            except Exception as e:
                self.logger.error(exc_info=e)

    def plot_multiple(self):
        os.makedirs(f"{self.folder}/figures", exist_ok=True)
        data = pd.DataFrame()
        for experimenter in self.experimenters:
            data_experiment = pd.DataFrame({
                "best_fitnesses":experimenter.quantumneat.best_fitnesses,
                "population_sizes":experimenter.quantumneat.population_sizes,  
                "number_of_species":experimenter.quantumneat.number_of_species, 
                "average_fitnesses":experimenter.quantumneat.average_fitnesses,
                "best_energies":experimenter.quantumneat.best_energies,
                })
            # self.logger.info(f"{data_experiment=}")
            data = pd.concat((data,data_experiment))
        final_energy_data = pd.DataFrame({"final_energies":experimenter.final_energies,})
        # self.logger.info(f"{data=}")
        sns.lineplot(data=data, x=data.index, y="best_fitnesses")
        # plt.legend()
        # plt.plot(self.quantumneat.best_fitnesses)
        plt.title("fitness_record")
        plt.xlabel("Generations")
        plt.savefig(f"{self.folder}/figures/{self.name}_multiple_runs_fitness_record.png")
        plt.close()
        sns.lineplot(data=data, x=data.index, y="population_sizes")
        plt.title("population_size")
        plt.xlabel("Generations")
        plt.savefig(f"{self.folder}/figures/{self.name}_multiple_runs_population_size.png")
        plt.close()
        sns.lineplot(data=data, x=data.index, y="number_of_species")
        plt.title("number_of_species")
        plt.xlabel("Generations")
        plt.savefig(f"{self.folder}/figures/{self.name}_multiple_runs_number_of_species.png")
        plt.close()
        sns.lineplot(data=data, x=data.index, y="average_fitnesses")
        plt.title("average_fitnesses")
        plt.xlabel("Generations")
        plt.savefig(f"{self.folder}/figures/{self.name}_multiple_runs_average_fitnesses.png")
        plt.close()
        sns.lineplot(data=data, x=data.index, y="best_energies")
        plt.title("best_energies")
        plt.xlabel("Generations")
        plt.ylabel("Energy")        
        plt.savefig(f"{self.folder}/figures/{self.name}_multiple_runs_best_energies.png")
        plt.close()
        # sns.histplot(data=final_energy_data, binwidth=1)
        sns.histplot(data=final_energy_data)
        plt.title("final_energies")
        plt.xlabel("Number of circuits in final population")
        plt.ylabel("Energy")
        plt.savefig(f"{self.folder}/figures/{self.name}_multiple_runs_final_energies.png")
        plt.close()