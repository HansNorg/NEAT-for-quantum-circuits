from __future__ import annotations

import os
from time import time
from abc import ABC, abstractmethod

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
        ("best_lengths", "Length of best circuit per generation", "#gates"),
        ("population_size", "Population size per generation", "Population"),
        ("number_of_species", "Number of species per generation", "Species"),
        ("average_fitnesses", "Average fitness per generation", "Fitness (a.u.)"),
        ("best_energies", "Best energy per generation", "Energy (a.u.)"),
        ("number_of_solutions", "Number of circuits that get \N{GREEK SMALL LETTER EPSILON} close to the real solution", "Circuits"),
        ("min_energies", "Lowest energy per generation", "Energy (a.u.)"),
    ]

class BasePlotter(ABC):
    def __init__(self, name:str, runs = "*", folder:str = ".", verbose = 0, error_verbose = 1) -> None:
        self.name = name
        self.runs = runs
        self.runs_name = "run" + runs
        self.folder = folder
        self.verbose = verbose
        self.error_verbose = error_verbose
        self.load_data()
        self.extra_title = ""

    @abstractmethod
    def load_data(self) -> None:
        self.data = dict()
        self.config = QuantumNEATConfig()
        self.generation_data = pd.DataFrame()
        self.evaluation_data = dict()
        self.evaluation_data_df = pd.DataFrame()

    def _plot_vs_generations(self, key:str, label:str=None, **plot_kwargs):
        try:
            sns.lineplot(data=self.generation_data, x="generation", y=key, label=label, **plot_kwargs)
        except ValueError as exc_info:
            if self.error_verbose == 1:
                print(f"{key} data not found for {self.name}")
            elif self.error_verbose >= 1:
                print(exc_info)
            return
        
    def plot_vs_generations(self, key:str, title:str, name:str, show=False, save=False):
        plt.figure()
        self._plot_vs_generations(key)
        self.finalise_plot(
            title=title, 
            xlabel="Generations", 
            ylabel=name, 
            savename=f"{self.runs_name}_{key}", 
            save=save, show=show,
            )
        
    def finalise_plot(self, *, title:str=None, xlabel:str=None,ylabel:str=None, legend:bool=False, savename:str="", save:bool=False, show:bool=False, close:bool=True) -> None:
        """
        Set basic plotstyle settings and save/show the plot.
        """
        if legend:
            plt.legend()
        plt.title(title)
        plt.grid()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if save:
            os.makedirs(f"{self.folder}/figures/{self.name}", exist_ok=True)
            plt.savefig(f"{self.folder}/figures/{self.name}/{savename}.png")
        if show:
            plt.show()
        elif close:
            plt.close()

    def _plot_evaluation(self, **plot_kwargs):
        sns.scatterplot(self.evaluation_data_df, x="distances", y="energies", **plot_kwargs)

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
        self.finalise_plot(
            title="Evaluation of best final circuit",
            xlabel="Distance (Angstrom)",
            ylabel="Energy (a.u.)",
            savename=f"{self.runs_name}_evaluation",
            save=save, show=show,
        )

    def _plot_delta_evaluation(self, absolute=False, **plot_kwargs):
        pass

    def plot_delta_evaluation(self, show = False, save = False, logarithmic=False, **plot_kwargs):
        if logarithmic:
            plt.yscale("log")
            self._plot_delta_evaluation(True, **plot_kwargs)
            logname = "_log"
            absname = "|"
        else:
            self._plot_delta_evaluation(**plot_kwargs)
            logname = ""
            absname = ""
        self.finalise_plot(
            title="Evaluation of best final circuit"+self.extra_title,
            xlabel="Distance (Angstrom)",
            ylabel=absname+"Delta energy (a.u.)"+absname,
            savename=f"{self.runs_name}_delta_evaluation{logname}",
            save=save, show=show,
        )

    def plot_all_generations(self, show = False, save = False):
        for key, title, name in GENERATION_DATA:
            self.plot_vs_generations(key, title+self.extra_title, name, show, save)
    
    def plot_all(self, show = False, save = False):
        self.plot_all_generations(show, save)
        self.plot_evaluation(show, save)
        self.plot_delta_evaluation(show, save)
        self.plot_delta_evaluation(show, save, logarithmic=True)
    
class SingleRunPlotter(BasePlotter):
    def load_data(self):
        self.data = dict(np.load(f"{self.folder}\\results\\{self.name}_run{self.runs}_results.npz", allow_pickle=True))
        # self.data_df = pd.DataFrame.from_dict(np.load(f"{self.folder}\\results\\{self.name}_run{self.runs}_results.npz", allow_pickle=True))
        self.generation_data = pd.DataFrame()
        for key, _, _ in GENERATION_DATA:
            try:
                self.generation_data[key] = pd.DataFrame(self.data.pop(key))
            except KeyError as exc_info:
                if self.error_verbose == 1:
                    print(f"{key} data not found for {self.name} run: {self.runs}")
                elif self.error_verbose >= 1:
                    print(exc_info)
                continue
        self.generation_data.index.name = "generation"
        self.config:QuantumNEATConfig = self.data.pop("config")

        try:
            evaluation_data = dict(np.load(f"{self.folder}\\results\\{self.name}_run{self.runs}_evaluation.npz", allow_pickle=True))
        except FileNotFoundError as exc_info:
            if self.error_verbose == 1:
                print(f"evaluation data not found")
            elif self.error_verbose >= 1:
                print(exc_info)
        self.evaluation_data = evaluation_data
        evaluation_data_df = pd.DataFrame()
        for key in ["distances", "energies"]:
            key_data = pd.DataFrame()
            for data in evaluation_data:
                key_data = pd.concat((key_data, pd.DataFrame(data[key])))
            if len(key_data) == 0:
                continue
            evaluation_data_df[key] = key_data
        self.evaluation_data_df = evaluation_data_df
        
    def _plot_species_evolution(self, sizes, colorscheme, line_color, n_generations, population_size, n_species):        
        if type(colorscheme) is str:
            colorscheme = [[colorscheme for _ in range(n_generations)] for _ in range(n_species)]
        
        for specie_ind in range(len(sizes)-1):
            for i in range(len(sizes[specie_ind])-1):
                plt.fill_between(x=[i+1, i+2], y1 = [sizes[specie_ind][i],sizes[specie_ind][i+1]], y2=[sizes[specie_ind+1][i], sizes[specie_ind+1][i+1]], color=colorscheme[specie_ind][i])

        for i in range(len(sizes)-1):
            plt.plot(range(1, n_generations+1), sizes[i+1], color=line_color)
        
        plt.xlim(1, n_generations)
        plt.ylim(0, population_size)
    
    def plot_species_evolution(self, show=False, save=False):
        try:
            species_data = self.data["species_data"]
        except Exception as exc_info:
            if self.error_verbose == 1:
                print(f"species_data data not found for {self.name}_run{self.runs}")
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
        self.finalise_plot(
            title = "Specie evolution over generations",
            xlabel= "Generation",
            ylabel= "Number of genomes in species",
            savename=f"run{self.runs}_species_evolution",
            save=save, show=show,
        )

        colormap = mpl.colormaps.get_cmap('gray')

        plt.figure()
        normalise = mplc.Normalize(vmin=avg_fitnesses.min(), vmax=avg_fitnesses.max())
        colorscheme = colormap(normalise(avg_fitnesses))
        self._plot_species_evolution(sizes, colorscheme, 'blue', n_generations, population_size, n_species)
        self.finalise_plot(
            title = "Specie evolution over generations",
            xlabel= "Generation",
            ylabel= "Number of genomes in species",
            savename=f"run{self.runs}_species_evolution_avg_fitness",
            save=save, show=show,
        )

        plt.figure()
        normalise = mplc.Normalize(vmin=best_fitnesses.min(), vmax=best_fitnesses.max())
        colorscheme = colormap(normalise(best_fitnesses))
        self._plot_species_evolution(sizes, colorscheme, 'blue', n_generations, population_size, n_species)
        self.finalise_plot(
            title = "Specie evolution over generations",
            xlabel= "Generation",
            ylabel= "Number of genomes in species",
            savename=f"run{self.runs}_species_evolution_best_fitness",
            save=save, show=show,
        )

    def plot_all(self, show = False, save = False):
        super().plot_all(show, save)
        self.plot_species_evolution(show, save)

    # def plot_evaluation(self, show = False, save = False):
    #     try:
    #         data = dict(np.load(f"{self.folder}\\results\\{self.name}_run{self.runs}_evaluation.npz", allow_pickle=True))
    #     except Exception as exc_info:
    #         if self.error_verbose == 1:
    #             print(f"evaluation data not found for {self.name}_run{self.runs}")
    #         elif self.error_verbose >= 1:
    #             print(exc_info)
    #         return
    #     if "gs" in self.name:
    #         if "h2" in self.name:
    #             molecule = "h2"
    #         elif "h6" in self.name:
    #             molecule = "h6"
    #         elif "lih" in self.name:
    #             molecule = "lih"
    #         else:
    #             molecule = None
    #         if molecule:
    #             gse = GroundStateEnergy(self.config, molecule)
    #             gse.plot_solution(color="r", linewidth=1, label="Solution (ED)")
    #     elif "h2" in self.name:
    #         plot_h2_solution(color="r", linewidth=1, label="Solution (ED)")
    #     elif "h6" in self.name:
    #         plot_h6_solution(color="r", linewidth=1, label="Solution (ED)")
    #     plt.scatter(data["distances"], data["energies"])
    #     self.finalise_plot(
    #         title="Evaluation of best final circuit",
    #         xlabel="Distance (Angstrom)",
    #         ylabel="Energy (a.u.)",
    #         savename=f"run{self.runs}_evaluation",
    #         save=save, show=show,
    #         )
        
    def _plot_delta_evaluation(self, absolute=False, **plot_kwargs):
        try:
            data = dict(np.load(f"{self.folder}\\results\\{self.name}_run{self.runs}_evaluation.npz", allow_pickle=True))
        except Exception as exc_info:
            if self.error_verbose == 1:
                print(f"evaluation data not found for {self.name}_run{self.runs}")
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
        if not molecule:
            return
        gse = GroundStateEnergy(self.config, molecule)
        difference = data["energies"]-gse.data["solution"]
        if absolute:
            difference = abs(difference)
        plt.scatter(data["distances"], difference, **plot_kwargs)

class MultipleRunPlotter(BasePlotter):
    def __init__(self, name: str, runs="*", folder: str = ".", verbose=0, error_verbose=1) -> None:
        super().__init__(name, runs, folder, verbose, error_verbose)
        self.extra_title = f" averaged over {self.n_runs} runs"
        self.runs_name = "multiple_runs"

    def load_data(self):
        # self.data = dict()
        self.generation_data = pd.DataFrame()
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
            try:
                data = dict(np.load(file, allow_pickle=True))
            except FileNotFoundError as exc_info:
                if self.error_verbose < 2:
                    print(f"{file} not found.")
                else:
                    print(exc_info)
                continue
            config:QuantumNEATConfig = data.pop("config")
            data_multiple.append(data)
        self.n_runs = len(data_multiple)
        if self.n_runs == 0:
            print(f"No files found for {self.name} runs {self.runs}")
            config = None
        self.config = config # All configs should be the same, so we can take only the last

        for key, _, _ in GENERATION_DATA:
            key_data = pd.DataFrame()
            for i, data in enumerate(data_multiple):
                try:
                    # print(data[key])
                    data_df = pd.DataFrame(data[key])
                except KeyError as exc_info:
                    if self.error_verbose == 1:
                        print(f"{key} data not found for the {i}th run of {self.name} runs: {self.runs}")
                    elif self.error_verbose >= 1:
                        print(exc_info)
                    continue
                key_data = pd.concat((key_data, data_df))
            if len(key_data) == 0:
                continue
            # self.data[key] = key_data
            self.generation_data[key] = key_data
        self.generation_data.index.name = "generation"
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
            except FileNotFoundError as exc_info:
                if self.error_verbose == 1:
                    print(f"{file} not found")
                elif self.error_verbose >= 1:
                    print(exc_info)
                continue
            evaluation_data.append(data)
        self.evaluation_data = evaluation_data
        evaluation_data_df = pd.DataFrame()
        for key in ["distances", "energies"]:
            key_data = pd.DataFrame()
            for data in evaluation_data:
                key_data = pd.concat((key_data, pd.DataFrame(data[key])))
            if len(key_data) == 0:
                continue
            evaluation_data_df[key] = key_data
        self.evaluation_data_df = evaluation_data_df

    def _plot_min_energy_single_point(self, x, c = None, **plot_kwargs):
        for data in self.evaluation_data:
            min_energy = data["energies"][0]
            plt.scatter(x, min_energy, c=c, **plot_kwargs)

    def _plot_diff_energy_single_point(self, x, solution, c = None, **plot_kwargs):
        for data in self.evaluation_data:
            min_energy = data["energies"][0]
            diff_energy = min_energy - solution
            plt.scatter(x, diff_energy, c=c, **plot_kwargs)

    # def _plot_evaluation(self, **plot_kwargs):
    #     for data in self.evaluation_data:
    #         plt.scatter(data["distances"], data["energies"], **plot_kwargs)

    def _plot_delta_evaluation(self, absolute = False, **plot_kwargs):
        if "gs" in self.name:
            if "h2" in self.name:
                molecule = "h2"
            elif "h6" in self.name:
                molecule = "h6"
            elif "lih" in self.name:
                molecule = "lih"
            else:
                molecule = None
        if not molecule:
            return
        gse = GroundStateEnergy(self.config, molecule)
        for data in self.evaluation_data:
            difference = data["energies"]-gse.data["solution"]
            if absolute:
                difference = abs(difference)
            plt.scatter(data["distances"], difference, **plot_kwargs)

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
        if not molecule:
            return energies
        gse = GroundStateEnergy(self.config, molecule)
        for data in self.evaluation_data:
            energies = pd.concat((energies, pd.DataFrame(data["energies"]-gse.data["solution"], index=data["distances"])))
        return energies

class MultipleExperimentPlotter(BasePlotter):
    def __init__(self, name: str, runs="*", folder: str = ".", verbose=0, error_verbose=1) -> None:
        super().__init__(name, runs, folder, verbose, error_verbose)
        self.experiments:list[tuple[MultipleRunPlotter, str]] = []
        self.extra_title = " multiple experiments"
        self.runs_name = "multiple_experiments"
    
    def load_data(self):
        pass

    def add_experiment(self, name, runs, label):
        self.experiments.append((MultipleRunPlotter(name, runs, self.folder, self.verbose, self.error_verbose), label))

    def add_experiments(self, experiments):
        for name, runs, label in experiments:
            self.add_experiment(name, runs, label)

    def _plot_vs_generations(self, key: str, label: str = None, **plot_kwargs):
        for i, (experiment, label) in enumerate(self.experiments):
            experiment._plot_vs_generations(key, label, **plot_kwargs)
    
    def _plot_min_energy_single_point(self, X, color = None, **plot_kwargs):
        for i, (experiment, label) in enumerate(self.experiments):
            experiment._plot_min_energy_single_point(X[i], color, **plot_kwargs)
    
    def plot_min_energy(self, X, title = None, show = False, save = False, **plot_kwargs):
        self._plot_min_energy_single_point(X, "b", **plot_kwargs)
        self.finalise_plot(
            title=title,
            xlabel="Distance between atoms (Angstrom)", #TODO angstrom symbol
            ylabel="Ground state energy (a.u.)",
            savename=f"energy_vs_R",
            save=save, show=show,
        )

    def _plot_diff_energy_single_point(self, X, solutions, color = None, **plot_kwargs):
        for i, (experiment, label) in enumerate(self.experiments):
            experiment._plot_diff_energy_single_point(X[i], solutions[i], color, **plot_kwargs)

    def plot_diff_energy(self, X, solutions, title = None, show = False, save = False, **plot_kwargs):
        self._plot_diff_energy_single_point(X, solutions, "b", **plot_kwargs)
        self.finalise_plot(
            title=title,
            xlabel="Distance between atoms (Angstrom)", #TODO angstrom symbol
            ylabel="Delta ground state energy (a.u.)",
            savename=f"delta_energy_vs_R",
            save=save, show=show,
        )

    def _plot_evaluation(self, colormap = None, **plot_kwargs):
        if colormap:
            n_experiments = len(self.experiments)
            colormap = mpl.colormaps.get_cmap(colormap).resampled(n_experiments)
            for i, (experiment, label) in enumerate(self.experiments):
                experiment._plot_evaluation(label = label, color=colormap(i/n_experiments), **plot_kwargs)    
            return
        for i, (experiment, label) in enumerate(self.experiments):
            experiment._plot_evaluation(label = label, **plot_kwargs)

    def _plot_delta_evaluation(self, colormap = None, absolute = False, **plot_kwargs):
        if colormap:
            n_experiments = len(self.experiments)
            colormap = mpl.colormaps.get_cmap(colormap).resampled(n_experiments)
            for i, (experiment, label) in enumerate(self.experiments):
                experiment._plot_delta_evaluation(label = label, absolute=absolute, color=colormap(i/n_experiments), **plot_kwargs)    
            return
        for i, (experiment, label) in enumerate(self.experiments):
            experiment._plot_delta_evaluation(label = label, absolute=absolute, **plot_kwargs)
    
    def get_energies(self):
        energies = pd.DataFrame()
        for experiment, name in self.experiments:
            energies[name] = experiment.get_energies()
        return energies
    
    def get_delta_energies(self):
        energies = pd.DataFrame()
        for experiment, name in self.experiments:
            energy = experiment.get_delta_energies()
            if len(energy) == 0:
                continue
            energies[name] = energy
        return energies
    
    def plot_box(self, xlabel, title = None, show=False, save=False, savename="", **plot_kwargs):
        energies = self.get_delta_energies()
        if len(energies) == 0:
            plt.close()
            return
        sns.boxplot(energies, **plot_kwargs)
        self.finalise_plot(
            title=title,
            xlabel=xlabel,
            ylabel="Delta energy",
            savename=f"delta_energy_boxplot{savename}",
            save=save, show=show,
        )

    def plot_box_abs(self, xlabel, title = None, show=False, save=False, savename="", **plot_kwargs):
        energies = self.get_delta_energies()
        if len(energies) == 0:
            plt.close()
            return
        sns.boxplot(abs(energies), **plot_kwargs)
        self.finalise_plot(
            title=title,
            xlabel=xlabel,
            ylabel="Delta energy",
            savename=f"delta_energy_boxplot{savename}",
            save=save, show=show,
        )

    def plot_box_log(self, xlabel, title = None, show=False, save=False, **plot_kwargs):
        plt.yscale("log")
        self.plot_box_abs(xlabel, title = title, show=show, save=save, savename="_log", **plot_kwargs)

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
            plotter = SingleRunPlotter(args.name, run, folder=args.folder, verbose=args.verbose, error_verbose=args.verbose)
            plotter.plot_all(show=args.show, save=args.save)
            # plotter.plot_species_evolution(show=args.show, save=args.save)
        else:
            plotter = MultipleRunPlotter(args.name, run, folder=args.folder, verbose=args.verbose, error_verbose=args.verbose)
            plotter.plot_all(show = args.show, save = args.save)