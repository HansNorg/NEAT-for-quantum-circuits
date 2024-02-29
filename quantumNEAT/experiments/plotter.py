from __future__ import annotations

import os
from time import time
from quantumneat.problems.chemistry import GroundStateEnergy
from quantumneat.problems.hydrogen import plot_solution as plot_h2_solution, get_solution as get_h2_solution, get_solutions as get_h2_solutions
from quantumneat.problems.hydrogen_6 import plot_solution as plot_h6_solution, get_solution as get_h6_solution, get_solutions as get_h6_solutions
from tqdm import tqdm
from pathlib import Path
from typing import TYPE_CHECKING
import warnings
warnings.filterwarnings("ignore", "use_inf_as_na")

import numpy as np
warnings.filterwarnings("ignore", category=np.ComplexWarning)
import matplotlib as mpl
import matplotlib.colors as mplc
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
    def __init__(self, name:str, run:int, folder:str = ".", verbose=0, error_verbose = 1) -> None:
        self.name = name
        self.run = run
        self.folder = folder
        self.verbose = verbose
        self.error_verbose = error_verbose
        self.load_data()
        
    def load_data(self):
        self.data = dict(np.load(f"{self.folder}\\results\\{self.name}_run{self.run}_results.npz", allow_pickle=True))
        self.config:QuantumNEATConfig = self.data.pop("config")

    def _plot_vs_generations(self, key:str, label:str = None):
        try:
            y = self.data[key]
        except Exception as exc_info:
            if self.error_verbose == 1:
                print(f"{key} data not found for {self.name}_run{self.run}")
            elif self.error_verbose >= 1:
                print(exc_info)
            return
        plt.plot(y, label=label)

    def plot_vs_generations(self, key:str, title:str, name:str, show=False, save=False):
        plt.figure()
        self._plot_vs_generations(key)
        plt.title(title)
        plt.grid()
        plt.xlabel("Generations")
        plt.ylabel(name)
        if save:
            os.makedirs(f"{self.folder}/figures/{self.name}", exist_ok=True)
            plt.savefig(f"{self.folder}\\figures\\{self.name}\\run{self.run}_{key}.png")
        if show:
            plt.show()
        plt.close()
        
    def _plot_species_evolution(self, sizes, colorscheme, line_color, n_generations, population_size, n_species):        
        if type(colorscheme) is str:
            colorscheme = [[colorscheme for _ in range(n_generations)] for _ in range(n_species)]
        
        for specie_ind in range(len(sizes)-1):
            for i in range(len(sizes[specie_ind])-1):
                plt.fill_between(x=[i+1, i+2], y1 = [sizes[specie_ind][i],sizes[specie_ind][i+1]], y2=[sizes[specie_ind+1][i], sizes[specie_ind+1][i+1]], color=colorscheme[specie_ind][i])

        for i in range(len(sizes)-1):
            plt.plot(range(1, n_generations+1), sizes[i+1], color=line_color)
        plt.title("Specie evolution over generations")
        plt.xlim(1, n_generations)
        plt.xlabel("Generation")
        plt.ylim(0, population_size)
        plt.ylabel("Number of genomes in species")
    
    def plot_species_evolution(self, show=False, save=False):
        try:
            species_data = self.data["species_data"]
        except Exception as exc_info:
            if self.error_verbose == 1:
                print(f"species_data data not found for {self.name}_run{self.run}")
            elif self.error_verbose >= 1:
                print(exc_info)
            return
        n_generations = int(species_data.T[0].max())
        n_species = int(species_data.T[1].max()) + 1

        sizes = np.zeros(shape=(n_generations, n_species+1))
        avg_fitnesses = np.zeros(shape=(n_generations, n_species))
        best_fitnesses = np.zeros(shape=(n_generations, n_species))
        for generation, specie, size, avg_fitness, best_fitness in species_data:
            sizes[int(generation)-1][int(specie)+1] = int(size)
            avg_fitnesses[int(generation)-1][int(specie)] = avg_fitness
            best_fitnesses[int(generation)-1][int(specie)] = best_fitness
        population_size = int(sum(sizes[0]))
        sizes = sizes.T
        avg_fitnesses = avg_fitnesses.T
        best_fitnesses = best_fitnesses.T
        for i in range(len(sizes)-1):
            sizes[i+1] += sizes[i]

        plt.figure()
        self._plot_species_evolution(sizes, 'black', 'white', n_generations, population_size, n_species)
        if save:
            os.makedirs(f"{self.folder}/figures/{self.name}", exist_ok=True)
            plt.savefig(f"{self.folder}\\figures\\{self.name}\\run{self.run}_species_evolution.png")
        if show:
            plt.show()
        plt.close()

        colormap = mpl.colormaps.get_cmap('gray')

        plt.figure()
        normalise = mplc.Normalize(vmin=avg_fitnesses.min(), vmax=avg_fitnesses.max())
        colorscheme = colormap(normalise(avg_fitnesses))
        self._plot_species_evolution(sizes, colorscheme, 'blue', n_generations, population_size, n_species)
        if save:
            os.makedirs(f"{self.folder}/figures/{self.name}", exist_ok=True)
            plt.savefig(f"{self.folder}\\figures\\{self.name}\\run{self.run}_species_evolution_avg_fitness.png")
        if show:
            plt.show()
        plt.close()

        plt.figure()
        normalise = mplc.Normalize(vmin=best_fitnesses.min(), vmax=best_fitnesses.max())
        colorscheme = colormap(normalise(best_fitnesses))
        self._plot_species_evolution(sizes, colorscheme, 'blue', n_generations, population_size, n_species)
        if save:
            os.makedirs(f"{self.folder}/figures/{self.name}", exist_ok=True)
            plt.savefig(f"{self.folder}\\figures\\{self.name}\\run{self.run}_species_evolution_best_fitness.png")
        if show:
            plt.show()
        plt.close()

    def plot_all(self, show = False, save = False):
        if save:
            os.makedirs(f"{self.folder}/figures/{self.name}", exist_ok=True)
        for key, title, name in GENERATION_DATA:
            self.plot_vs_generations(key, title, name, show, save)
        self.plot_species_evolution(show, save)        
        self.plot_evaluation(show, save)
        if "h2" in self.name:
            self.plot_delta_evaluation(get_h2_solutions, show, save)
        elif "h6" in self.name:
            self.plot_delta_evaluation(get_h6_solutions, show, save)
        self.plot_delta_evaluation_new(show, save)

    def _plot_min_energy_single_point(self, x, color = None):
        try:
            min_energy = min(self.data["min_energies"])
        except Exception as exc_info:
            if self.error_verbose == 1:
                print(f"min_energies data not found for {self.name}_run{self.run}")
            elif self.error_verbose >= 1:
                print(exc_info)
            return
        plt.scatter(x, min_energy, c=color)

    def _plot_diff_energy_single_point(self, x, solution, color = None):
        try:
            min_energy = min(self.data["min_energies"])
        except Exception as exc_info:
            if self.error_verbose == 1:
                print(f"min_energies data not found for {self.name}_run{self.run}")
            elif self.error_verbose >= 1:
                print(exc_info)
            return
        diff_energy = min_energy - solution
        plt.scatter(x, diff_energy, c=color)

    def plot_evaluation(self, show = False, save = False):
        try:
            data = dict(np.load(f"{self.folder}\\results\\{self.name}_run{self.run}_evaluation.npz", allow_pickle=True))
        except Exception as exc_info:
            if self.error_verbose == 1:
                print(f"evaluation data not found for {self.name}_run{self.run}")
            elif self.error_verbose >= 1:
                print(exc_info)
            return
        if "gs" in self.name:
            if "h2" in self.name:
                molecule = "h2"
            elif "h6" in self.name:
                molecule = "h6"
            elif "lih" in self.name:
                molecule = "lih"
            else:
                molecule = None
            if molecule:
                gse = GroundStateEnergy(self.config, molecule)
                gse.plot_solution(color="r", linewidth=1, label="Solution (ED)")
        elif "h2" in self.name:
            plot_h2_solution(color="r", linewidth=1, label="Solution (ED)")
        elif "h6" in self.name:
            plot_h6_solution(color="r", linewidth=1, label="Solution (ED)")
        plt.scatter(data["distances"], data["energies"])
        plt.title("Evaluation of best final circuit")
        plt.xlabel("Distance (Angstrom)")
        plt.ylabel("Energy (a.u.)")
        if save:
            os.makedirs(f"{self.folder}/figures/{self.name}", exist_ok=True)
            plt.savefig(f"{self.folder}\\figures\\{self.name}\\run{self.run}_evaluation.png")
        if show:
            plt.show()
        plt.close()

    def _plot_delta_evaluation(self, solution_func, **plot_kwargs):
        try:
            data = dict(np.load(f"{self.folder}\\results\\{self.name}_run{self.run}_evaluation.npz", allow_pickle=True))
        except Exception as exc_info:
            if self.error_verbose == 1:
                print(f"evaluation data not found for {self.name}_run{self.run}")
            elif self.error_verbose >= 1:
                print(exc_info)
            return
        solutions = solution_func(data["distances"])
        plt.scatter(data["distances"], data["energies"]-solutions, **plot_kwargs)

    def plot_delta_evaluation(self, solution_func, show = False, save = False):
        self._plot_delta_evaluation(solution_func)
        plt.title("Evaluation of best final circuit")
        plt.xlabel("Distance (Angstrom)")
        plt.ylabel("Delta energy (a.u.)")
        if save:
            os.makedirs(f"{self.folder}/figures/{self.name}", exist_ok=True)
            plt.savefig(f"{self.folder}\\figures\\{self.name}\\run{self.run}_delta_evaluation.png")
        if show:
            plt.show()
        plt.close()
        
    def _plot_delta_evaluation_new(self, **plot_kwargs):
        try:
            data = dict(np.load(f"{self.folder}\\results\\{self.name}_run{self.run}_evaluation.npz", allow_pickle=True))
        except Exception as exc_info:
            if self.error_verbose == 1:
                print(f"evaluation data not found for {self.name}_run{self.run}")
            elif self.error_verbose >= 1:
                print(exc_info)
            return
        if "gs" in self.name:
            if "h2" in self.name:
                molecule = "h2"
            elif "h6" in self.name:
                molecule = "h6"
            elif "lih" in self.name:
                molecule = "lih"
            else:
                molecule = None
            if molecule:
                gse = GroundStateEnergy(self.config, molecule)
                plt.scatter(data["distances"], data["energies"]-gse.data["solution"], **plot_kwargs)

    def plot_delta_evaluation_new(self, show = False, save = False):
        self._plot_delta_evaluation_new()
        plt.title("Evaluation of best final circuit")
        plt.xlabel("Distance (Angstrom)")
        plt.ylabel("Delta energy (a.u.)")
        if save:
            os.makedirs(f"{self.folder}/figures/{self.name}", exist_ok=True)
            plt.savefig(f"{self.folder}\\figures\\{self.name}\\run{self.run}_delta_evaluation.png")
        if show:
            plt.show()
        plt.close()

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

        evaluation_data = []
        if self.runs == "*":
            files = Path(f"{self.folder}\\results\\").glob(f"{self.name}_run{self.runs}_evaluation.npz")
        else:
            files = [
                f"{self.folder}\\results\\{self.name}_run{run}_evaluation.npz"
                for run in eval(self.runs)
                ]
        for file in files:
            if self.verbose >= 1:
                print(file)
            try:
                data = dict(np.load(file, allow_pickle=True))
            except Exception as exc_info:
                if self.error_verbose == 1:
                    print(f"evaluation data not found for {self.name}_run{run}")
                elif self.error_verbose >= 1:
                    print(exc_info)
                continue
            evaluation_data.append(data)
        self.evaluation_data = evaluation_data
    
    def _plot_vs_generations(self, key:str, label:str=None):
        try:
            data:pd.DataFrame = self.data[key]
        except Exception as exc_info:
            if self.error_verbose == 1:
                print(f"{key} data not found for {self.name}")
            elif self.error_verbose >= 1:
                print(exc_info)
            return
        y = np.real(data[0])
        sns.lineplot(data=data, x=data.index, y=y, label=label)

    def plot_all(self, show = False, save = False):
        extra_title = f" averaged over {self.n_runs} runs"
        for key, title, name in GENERATION_DATA:
            plt.figure()
            self._plot_vs_generations(key)
            plt.title(title+extra_title)
            plt.grid()
            plt.xlabel("Generations")
            plt.ylabel(name)
            if save:
                os.makedirs(f"{self.folder}/figures/{self.name}", exist_ok=True)
                plt.savefig(f"{self.folder}\\figures\\{self.name}\\multiple_runs_{key}.png")
            if show:
                plt.show()
            plt.close()
        self.plot_evaluation(show, save)
        self.plot_delta_evaluation(show, save)
        self.plot_delta_evaluation(show, save, logarithmic=True)
        
    def _plot_min_energy_single_point(self, x, c = None, **plot_kwargs):
        for data in self.evaluation_data:
            min_energy = data["energies"][0]
            plt.scatter(x, min_energy, c=c, **plot_kwargs)

    def _plot_diff_energy_single_point(self, x, solution, c = None, **plot_kwargs):
        for data in self.evaluation_data:
            min_energy = data["energies"][0]
            diff_energy = min_energy - solution
            plt.scatter(x, diff_energy, c=c, **plot_kwargs)

    def _plot_evaluation(self, **plot_kwargs):
        for data in self.evaluation_data:
            plt.scatter(data["distances"], data["energies"], **plot_kwargs)
    
    def plot_evaluation(self, show=False, save=False, **plot_kwargs):
        if "gs" in self.name:
            if "h2" in self.name:
                molecule = "h2"
            elif "h6" in self.name:
                molecule = "h6"
            elif "lih" in self.name:
                molecule = "lih"
            else:
                molecule = None
            if molecule:
                gse = GroundStateEnergy(self.config, molecule)
                gse.plot_solution(color="r", linewidth=1, label="Solution (ED)")
        elif "h2" in self.name:
            plot_h2_solution(color="r", linewidth=1, label="Solution (ED)")
        elif "h6" in self.name:
            plot_h6_solution(color="r", linewidth=1, label="Solution (ED)")
        self._plot_evaluation(**plot_kwargs)
        plt.title("Evaluation of best final circuit")
        plt.xlabel("Distance (Angstrom)")
        plt.ylabel("Energy (a.u.)")
        if save:
            os.makedirs(f"{self.folder}/figures/{self.name}", exist_ok=True)
            plt.savefig(f"{self.folder}\\figures\\{self.name}\\multiple_runs_evaluation.png")
        if show:
            plt.show()
        plt.close()

    def _plot_delta_evaluation(self, solution_func, **plot_kwargs):
        for data in self.evaluation_data:
            solutions = solution_func(data["distances"])
            plt.scatter(data["distances"], data["energies"]-solutions, **plot_kwargs)

    def plot_delta_evaluation(self, show=False, save=False, logarithmic=False, **plot_kwargs):
        if "h2" in self.name:
            self._plot_delta_evaluation(get_h2_solutions, **plot_kwargs)
        elif "h6" in self.name:
            self._plot_delta_evaluation(get_h6_solutions, **plot_kwargs)
        else:
            return
        logname = ""
        if logarithmic:
            plt.yscale("log")
            logname = "_logarithmic"
        plt.title("Evaluation of best final circuit")
        plt.xlabel("Distance (Angstrom)")
        plt.ylabel("Delta energy (a.u.)")
        if save:
            os.makedirs(f"{self.folder}/figures/{self.name}", exist_ok=True)
            plt.savefig(f"{self.folder}\\figures\\{self.name}\\multiple_runs_delta_evaluation{logname}.png")
        if show:
            plt.show()
        plt.close()

    def _plot_delta_evaluation_new(self, **plot_kwargs):
        if "gs" in self.name:
            if "h2" in self.name:
                molecule = "h2"
            elif "h6" in self.name:
                molecule = "h6"
            elif "lih" in self.name:
                molecule = "lih"
            else:
                molecule = None
        for data in self.evaluation_data:
            if molecule:
                gse = GroundStateEnergy(self.config, molecule)
                plt.scatter(data["distances"], data["energies"]-gse.data["solution"], **plot_kwargs)

    def plot_delta_evaluation_new(self, show = False, save = False):
        self._plot_delta_evaluation_new()
        plt.title("Evaluation of best final circuit")
        plt.xlabel("Distance (Angstrom)")
        plt.ylabel("Delta energy (a.u.)")
        if save:
            os.makedirs(f"{self.folder}/figures/{self.name}", exist_ok=True)
            plt.savefig(f"{self.folder}\\figures\\{self.name}\\run{self.run}_delta_evaluation.png")
        if show:
            plt.show()
        plt.close()

    def get_energies(self) -> pd.DataFrame:
        energies = pd.DataFrame()
        for data in self.evaluation_data:
            energies = pd.concat((energies, pd.DataFrame(data["energies"], index=data["distances"])))
        return energies
    
    def get_delta_energies(self) -> pd.DataFrame:
        energies = pd.DataFrame()
        if "gs" in self.name:
            if "h2" in self.name:
                molecule = "h2"
            elif "h6" in self.name:
                molecule = "h6"
            elif "lih" in self.name:
                molecule = "lih"
            else:
                molecule = None
        for data in self.evaluation_data:
            if molecule:
                gse = GroundStateEnergy(self.config, molecule)
                energies = pd.concat((energies, pd.DataFrame(data["energies"]-gse.data["solution"], index=data["distances"])))
        return energies

class MultipleExperimentPlotter:
    def __init__(self,name:str, folder:str = ".", verbose = 0, error_verbose = 1) -> None:
        self.name = name
        self.folder = folder
        self.verbose = verbose
        self.error_verbose = error_verbose
        self.experiments:list[tuple[MultipleRunPlotter, str]] = []
    
    def add_experiment(self, name, runs, label):
        self.experiments.append((MultipleRunPlotter(name, runs, self.folder, self.verbose, self.error_verbose), label))

    def add_experiments(self, experiments):
        for name, runs, label in experiments:
            self.add_experiment(name, runs, label)
        
    def plot_all(self, show=False, save=False):
        extra_title = f" multiple experiments"
        if save:
            os.makedirs(f"{self.folder}/figures/{self.name}", exist_ok=True)
        for key, title, name in GENERATION_DATA:
            plt.figure()
            for experiment, label in self.experiments:
                experiment._plot_vs_generations(key, label=f"{label}: {experiment.n_runs}")
            plt.title(title+extra_title)
            plt.grid()
            plt.xlabel("Generations")
            plt.ylabel(name)
            if save:
                plt.savefig(f"{self.folder}\\figures\\{self.name}\\{key}.png")
            if show:
                plt.show()
            plt.close()
        # if "h2" in self.name:
        #     X, solutions = get_h2_solution()
        #     self.plot_diff_energy(X, solutions, "Difference from solution", show, save)

    def _plot_min_energy_single_point(self, X, color = None, **plot_kwargs):
        for i, (experiment, label) in enumerate(self.experiments):
            experiment._plot_min_energy_single_point(X[i], color, **plot_kwargs)
    
    def plot_min_energy(self, X, title = None, show = False, save = False, **plot_kwargs):
        self._plot_min_energy_single_point(X, "b", **plot_kwargs)
        plt.title(title)
        plt.grid()
        plt.xlabel("Distance between atoms (Angstrom)") #TODO angstrom symbol
        plt.ylabel("Ground state energy (a.u.)")
        if save:
            os.makedirs(f"{self.folder}/figures/{self.name}", exist_ok=True)
            plt.savefig(f"{self.folder}\\figures\\{self.name}\\energy_vs_R.png")
        if show:
            plt.show()
        plt.close()

    def _plot_diff_energy_single_point(self, X, solutions, color = None, **plot_kwargs):
        for i, (experiment, label) in enumerate(self.experiments):
            experiment._plot_diff_energy_single_point(X[i], solutions[i], color, **plot_kwargs)

    def plot_diff_energy(self, X, solutions, title = None, show = False, save = False, **plot_kwargs):
        self._plot_diff_energy_single_point(X, solutions, "b", **plot_kwargs)
        plt.title(title)
        plt.grid()
        plt.xlabel("Distance between atoms (Angstrom)") #TODO angstrom symbol
        plt.ylabel("Ground state energy (a.u.)")
        if save:
            os.makedirs(f"{self.folder}/figures/{self.name}", exist_ok=True)
            plt.savefig(f"{self.folder}\\figures\\{self.name}\\delta_energy_vs_R.png")
        if show:
            plt.show()
        plt.close()

    def _plot_evaluation(self, **plot_kwargs):
        for i, (experiment, label) in enumerate(self.experiments):
            experiment._plot_evaluation(label = label, **plot_kwargs)

    def plot_evaluation(self, title = None, show = False, save = False, **plot_kwargs):
        self._plot_evaluation(**plot_kwargs)
        plt.title(title)
        plt.grid()
        plt.legend()
        plt.xlabel("Distance between atoms (Angstrom)") #TODO angstrom symbol
        plt.ylabel("Ground state energy (a.u.)")
        if save:
            os.makedirs(f"{self.folder}/figures/{self.name}", exist_ok=True)
            plt.savefig(f"{self.folder}\\figures\\{self.name}\\evaluation.png")
        if show:
            plt.show()
        plt.close()

    def _plot_delta_evaluation(self, solution_func, **plot_kwargs):
        for i, (experiment, label) in enumerate(self.experiments):
            experiment._plot_delta_evaluation(solution_func, label = label, **plot_kwargs)

    def plot_delta_evaluation(self, solution_func, title = None, show = False, save = False, savename = "delta_evaluation", **plot_kwargs):
        self._plot_delta_evaluation(solution_func, **plot_kwargs)
        plt.title(title)
        plt.grid()
        plt.legend()
        plt.xlabel("Distance between atoms (Angstrom)") #TODO angstrom symbol
        plt.ylabel("Delta energy (a.u.)")
        if save:
            os.makedirs(f"{self.folder}/figures/{self.name}", exist_ok=True)
            plt.savefig(f"{self.folder}\\figures\\{self.name}\\{savename}.png")
        if show:
            plt.show()
        plt.close()

    def _plot_delta_evaluation_new(self, **plot_kwargs):
        for i, (experiment, label) in enumerate(self.experiments):
            experiment._plot_delta_evaluation_new(label = label, **plot_kwargs)

    def plot_delta_evaluation_new(self, title = None, show = False, save = False, savename = "delta_evaluation", **plot_kwargs):
        self._plot_delta_evaluation_new(**plot_kwargs)
        plt.title(title)
        plt.grid()
        plt.legend()
        plt.xlabel("Distance between atoms (Angstrom)") #TODO angstrom symbol
        plt.ylabel("Delta energy (a.u.)")
        if save:
            os.makedirs(f"{self.folder}/figures/{self.name}", exist_ok=True)
            plt.savefig(f"{self.folder}\\figures\\{self.name}\\{savename}.png")
        if show:
            plt.show()
        plt.close()
    
    def plot_delta_evaluation_log(self, solution_func, title = None, show = False, save = False, **plot_kwargs):
        plt.yscale("log")
        self.plot_delta_evaluation(solution_func, title, show, save, "delta_evaluation_log", **plot_kwargs)

    def plot_delta_evaluation_new_log(self, title = None, show = False, save = False, **plot_kwargs):
        plt.yscale("log")
        self.plot_delta_evaluation_new(title, show, save, "delta_evaluation_log", **plot_kwargs)

    def get_energies(self):
        energies = pd.DataFrame()
        for experiment, name in self.experiments:
            energies[name] = experiment.get_energies()
        return energies
    
    def get_delta_energies(self):
        energies = pd.DataFrame()
        for experiment, name in self.experiments:
            energies[name] = experiment.get_delta_energies()
        return energies
    
    def plot_box(self, xlabel, title = None, show=False, save=False, savename="", **plot_kwargs):
        energies = self.get_delta_energies()
        sns.boxplot(energies, **plot_kwargs)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Delta energy")
        if save:
            os.makedirs(f"{self.folder}/figures/{self.name}", exist_ok=True)
            plt.savefig(f"{self.folder}\\figures\\{self.name}\\delta_energy_boxplot{savename}.png")
        if show:
            plt.show()
        plt.close()

    def plot_box_log(self, xlabel, title = None, show=False, save=False, **plot_kwargs):
        plt.yscale("log")
        self.plot_box(xlabel, title = title, show=show, save=save, savename="_log", **plot_kwargs)

if __name__ == "__main__":
    from argparse import ArgumentParser
    argparser = ArgumentParser()
    # argparser.add_argument("problem",                   type=str,                                     help="which problem to plot")
    # argparser.add_argument("implementation",            type=str, choices=["linear_growth", "qneat"], help="which implementation was used")
    # argparser.add_argument("--name",                    type=str,                                     help="experiment name")
    argparser.add_argument("name", type=str, help="What to plot")
    argparser.add_argument("run", action="extend", nargs="+", help="Which run(s) to plot")
    argparser.add_argument("--cluster", dest='folder', nargs='?', default=".", const=".\\cluster")
    argparser.add_argument('--verbose', '-v', action='count', default=0)
    argparser.add_argument('--show', action="store_true")
    argparser.add_argument('--save', action="store_true")
    args = argparser.parse_args()
    for run in args.run:
        if run.isdigit():
            plotter = SingleRunPlotter(args.name, run, folder=args.folder, verbose=args.verbose, error_verbose=1)
            plotter.plot_all(show=args.show, save=args.save)
            # plotter.plot_species_evolution(show=args.show, save=args.save)
        else:
            plotter = MultipleRunPlotter(args.name, run, folder=args.folder, verbose=args.verbose, error_verbose=1)
            plotter.plot_all(show = args.show, save = args.save)