from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
import warnings
warnings.filterwarnings("ignore", "use_inf_as_na")

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns


if TYPE_CHECKING:
    from quantumneat.configuration import QuantumNEATConfig

GENERATION_DATA = [
        ("fitness_record", "Best fitness per generation", "Fitness (a.u.)"),
        ("population_size", "Population size per generation", "Population"),
        ("number_of_species", "Number of species per generation", "Species"),
        ("average_fitnesses", "Average fitness per generation", "Fitness (a.u.)"),
        ("best_energies", "Best energy per generation", "Energy (a.u.)"),
        ("number_of_solutions", "Number of circuits that get \N{GREEK SMALL LETTER EPSILON} close to the real solution", "Circuits"),
        ("min_energies", "Lowest energy per generation", "Energy (a.u.)"),
    ]

class SingleRunPlotter:
    def __init__(self, name:str, run:int, folder:str = ".", error_verbose = 1) -> None:
        self.name = name
        self.run = run
        self.folder = folder
        self.error_verbose = error_verbose
        self.load_data()
        
    def load_data(self):
        self.data = dict(np.load(f"{self.folder}\\results\\{self.name}_run{self.run}_results.npz", allow_pickle=True))
        self.config:QuantumNEATConfig = self.data.pop("config")
    
    def plot_vs_generations(self, key:str, label:str = None):
        try:
            y = self.data[key]
        except Exception as exc_info:
            if self.error_verbose == 1:
                print(f"{key} data not found for {self.name}_run{self.run}")
            elif self.error_verbose >= 1:
                print(exc_info)
            return
        plt.plot(y, label=label)

    def plot_species_contour(self):
        try:
            specie_sizes = self.data["specie_sizes"]
        except Exception as exc_info:
            if self.error_verbose == 1:
                print(f"specie_sizes data not found for {self.name}_run{self.run}")
            elif self.error_verbose >= 1:
                print(exc_info)
            return
        print(specie_sizes)

    def plot_all(self, show = False, save = False):
        if save:
            os.makedirs(f"{self.folder}/figures/{self.name}", exist_ok=True)
        # for key, title, name in GENERATION_DATA:
        #     plt.figure()
        #     self.plot_vs_generations(key)
        #     plt.title(title)
        #     plt.xlabel("Generations")
        #     plt.ylabel(name)
        #     if show:
        #         plt.show()
        #     if save:
        #         plt.savefig(f"{self.folder}\\figures\\{self.name}\\run{self.run}_{key}.png")
        #     plt.close()
        plt.figure()
        self.plot_species_contour()
        plt.title("Specie evolution over generations")
        plt.xlabel("Specie")
        plt.ylabel("Generations")
        if show:
            plt.show()
        if save:
            plt.savefig(f"{self.folder}\\figures\\{self.name}\\run{self.run}_specie_contour.png")


class MultipleRunPlotter:
    def __init__(self, name:str, runs = "*", folder:str = ".", verbose = 0, error_verbose = 1) -> None:
        self.name = name
        self.runs = runs
        self.folder = folder
        self.verbose = verbose
        self.error_verbose = error_verbose
        self.load_data()
        
    def load_data(self):
        self.data = dict()
        data_multiple = []
        if self.runs == "*":
            files = Path(f"{self.folder}\\results\\").glob(f"{self.name}_run{self.runs}_results.npz")
        else:
            files = [
                f"{self.folder}\\results\\{self.name}_run{run}_results.npz"
                for run in eval(self.runs)
                ]
        for file in files:
            if self.verbose >= 1:
                print(file)
            data = dict(np.load(file, allow_pickle=True))
            config:QuantumNEATConfig = data.pop("config")
            data_multiple.append(data)
        self.config = config # All configs should be the same, so we can take only the last
        self.n_runs = len(data_multiple)

        for key, _, _ in GENERATION_DATA:
            key_data = pd.DataFrame()
            for data in data_multiple:
                # print(data[key])
                data = pd.DataFrame(data[key])
                key_data = pd.concat((key_data, data))
            self.data[key] = key_data
        # print(self.data)
        # print(self.data[GENERATION_DATA[0][0]].head())
    
    def plot_vs_generations(self, key:str, label:str=None):
        try:
            data:pd.DataFrame = self.data[key]
        except Exception as exc_info:
            if self.error_verbose == 1:
                print(f"{key} data not found for {self.name}")
            elif self.error_verbose >= 1:
                print(exc_info)
            return
        sns.lineplot(data=data, x=data.index, y=data[0], label=label)

    def plot_all(self, show = False, save = False):
        extra_title = f" averaged over {self.n_runs} runs"
        for key, title, name in GENERATION_DATA:
            plt.figure()
            self.plot_vs_generations(key)
            plt.title(title+extra_title)
            plt.xlabel("Generations")
            plt.ylabel(name)
            if show:
                plt.show()
            if save:
                os.makedirs(f"{self.folder}/figures/{self.name}", exist_ok=True)
                plt.savefig(f"{self.folder}\\figures\\{self.name}\\multiple_runs_{key}.png")
            plt.close()

class MultipleExperimentPlotter:
    def __init__(self,name:str, folder:str = ".", verbose = 0, error_verbose = 1) -> None:
        self.name = name
        self.folder = folder
        self.verbose = verbose
        self.error_verbose = error_verbose
        self.experiments:list[tuple[MultipleRunPlotter, str]] = []
    
    def add_experiment(self, name, runs, label):
        self.experiments.append((MultipleRunPlotter(name, runs, self.folder, self.verbose, self.error_verbose), label))

    def add_experiments(self, experiments, runs):
        for name, label in experiments:
            self.add_experiment(name, runs, label)
        
    def plot_all(self, show=False, save=False):
        extra_title = f" multiple experiments"
        for key, title, name in GENERATION_DATA:
            plt.figure()
            for experiment, label in self.experiments:
                experiment.plot_vs_generations(key, label=f"{label}: {experiment.n_runs}")
            plt.title(title+extra_title)
            plt.xlabel("Generations")
            plt.ylabel(name)
            if show:
                plt.show()
            if save:
                os.makedirs(f"{self.folder}/figures/{self.name}", exist_ok=True)
                plt.savefig(f"{self.folder}\\figures\\{self.name}\\{key}.png")
            plt.close()

if __name__ == "__main__":
    from argparse import ArgumentParser
    argparser = ArgumentParser()
    # argparser.add_argument("problem",                   type=str,                                     help="which problem to plot")
    # argparser.add_argument("implementation",            type=str, choices=["linear_growth", "qneat"], help="which implementation was used")
    # argparser.add_argument("--name",                    type=str,                                     help="experiment name")
    argparser.add_argument("name", type=str, help="What to plot")
    argparser.add_argument("run", help="Which run(s) to plot")
    args = argparser.parse_args()
    if args.run.isdigit():
        plotter = SingleRunPlotter(args.name, args.run, error_verbose=1)
    else:
        plotter = MultipleRunPlotter(args.name, args.run, error_verbose=1)
        plotter.plot_all(show=True)            plotter.plot_species_contour()
        else:
