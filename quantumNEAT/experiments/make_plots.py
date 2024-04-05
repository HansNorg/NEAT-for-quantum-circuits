import os
from tqdm import tqdm 
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from experiments.plotter import MultipleExperimentPlotter
from quantumneat.problems.chemistry import GroundStateEnergy
from quantumneat.problems.hydrogen import get_solutions, plot_solution as plot_h2_solution, plot_UCCSD_diff as plot_UCCSD_diff_h2, plot_UCCSD_result as plot_UCCSD_result_h2
from experiments.run_experiment import cluster_n_shots

UCCSD_COLOR = "black"
HE_COLOR = "magenta"

def constant_population_size(folder, verbose, show = False, save = False):
    if verbose >= 1:
        print("Constant population size")
    plotter = MultipleExperimentPlotter("constant_population_size", folder=folder, verbose=verbose)
    experiments = [
        ("tfim_linear_growth_ROT-CNOT_5-qubits_100-population_100-optimizer-steps_no-forced-population", "*", "original"),
        ("tfim_linear_growth_ROT-CNOT_5-qubits_100-population_100-optimizer-steps_no-forced-population_normalised-fitness", "*", "normalised-fitness"),
        ("tfim_linear_growth_ROT-CNOT_5-qubits_100-population_100-optimizer-steps", 'range(23, 33)', "forced_population"),
        ("tfim_linear_growth_ROT-CNOT_5-qubits_100-population_100-optimizer-steps_normalised-fitness", "*", "forced_population_normalised-fitness"),
    ]
    plotter.add_experiments(experiments)
    plotter.plot_all(show, save)

def stagnant_experiment(folder, verbose, show = False, save = False):
    if verbose >= 1:
        print("Stagnant experiment")
    plotter = MultipleExperimentPlotter("stagnant_experiment", folder=folder, verbose=verbose)
    experiments = [
        ("tfim_linear_growth_ROT-CNOT_5-qubits_100-population_100-optimizer-steps_remove-stagnant", "*", "Stagnant removed"), # Part is 100, part is 1000 generations
        ("tfim_linear_growth_ROT-CNOT_5-qubits_100-population_100-optimizer-steps", str(list(range(33, 53))), "No removal"), # Part is 100, part is 1000 generations
    ]
    plotter.add_experiments(experiments)
    plotter.plot_all(show, save)

    plotter = MultipleExperimentPlotter("stagnant_experiment_separate", folder=folder, verbose=verbose)
    experiments = [
        ("tfim_linear_growth_ROT-CNOT_5-qubits_100-population_100-optimizer-steps_remove-stagnant", "[0, 2, 3, 4, 5, 6, 7, 8, 9, 10]", "Stagnant removed"), # 100 generations
        ("tfim_linear_growth_ROT-CNOT_5-qubits_100-population_100-optimizer-steps_remove-stagnant", "[1, 11, 12, 13, 14, 15, 16, 17]", "Stagnant removed"), # 1000 generations
        ("tfim_linear_growth_ROT-CNOT_5-qubits_100-population_100-optimizer-steps", "[33, 35, 36, 37, 38, 39, 40, 41, 43, 44]", "No removal"), # 100 generations
        ("tfim_linear_growth_ROT-CNOT_5-qubits_100-population_100-optimizer-steps", "[34, 42, 45, 46, 47, 48, 49, 50, 51, 52]", "No removal"), # 1000 generations
    ]
    plotter.add_experiments(experiments)
    plotter.plot_all(show, save)

def optimizer_steps(folder, verbose, show = False, save = False):
    if verbose >= 1:
        print("Optimizer steps")
    plotter = MultipleExperimentPlotter("optimizer_steps", folder=folder, verbose=verbose)
    experiments = [
        ("tfim_linear_growth_ROT-CNOT_5-qubits_100-population_100-optimizer-steps", str(list(range(33, 53))), "100 steps"), # Part is 100, part is 1000 generations
        ("tfim_linear_growth_ROT-CNOT_5-qubits_100-population_200-optimizer-steps", "*", "200 steps"), # Part is 100, part is 1000 generations
    ]
    plotter.add_experiments(experiments)
    plotter.plot_all(show, save)

    plotter = MultipleExperimentPlotter("optimizer_steps_separate", folder=folder, verbose=verbose)
    experiments = [
        ("tfim_linear_growth_ROT-CNOT_5-qubits_100-population_100-optimizer-steps", "[33, 35, 36, 37, 38, 39, 40, 41, 43, 44]", "100 steps"), # 100 generations
        ("tfim_linear_growth_ROT-CNOT_5-qubits_100-population_100-optimizer-steps", "[34, 42, 45, 46, 47, 48, 49, 50, 51, 52]", "100 steps"), # 1000 generations
        ("tfim_linear_growth_ROT-CNOT_5-qubits_100-population_200-optimizer-steps", "[0, 2, 3, 4, 5, 6, 7, 8, 9, 10]", "200 steps"), # 100 generations
        ("tfim_linear_growth_ROT-CNOT_5-qubits_100-population_200-optimizer-steps", "[1, 11, 12, 13, 14, 15, 16, 17, 18]", "200 steps"), # 1000 generations
    ]
    plotter.add_experiments(experiments)
    plotter.plot_all(show, save)

def hydrogen_atom_separate(folder, verbose, show = False, save = False):
    if verbose >= 1:
        print("Hydrogen atom")
    plotter = MultipleExperimentPlotter("hydrogen_atom_separate", folder=folder, verbose=verbose, error_verbose=verbose)
    # distances = [0.2, 0.35, 0.45, 1.5, 2.8]
    # experiments = [(f"h2_r_{R}_linear_growth_ROT-CNOT_2-qubits_100-population_100-optimizer-steps", "[1]", f"R = {R}") for R in distances]
    distances = np.arange(0.2, 2.90, 0.05)
    experiments = [(f"h2_r_{R:.2f}_linear_growth_ROT-CNOT_2-qubits_100-population_100-optimizer-steps", "[1]", f"R = {R:.2f}") for R in distances]
    # print(experiments)
    plotter.add_experiments(experiments)
    plotter.plot_all(show, save)
    plot_h2_solution(color="r", linewidth=1)
    plotter.plot_min_energy(distances, "Hydrogen atom", show, save, marker="x")
    # plotter.plot_min_energy(distances, "Hydrogen atom", show, save, marker="x", zorder=2)

def hydrogen_atom(folder, verbose, show = False, save = False):
    if verbose >= 1:
        print("Hydrogen atom")
    plotter = MultipleExperimentPlotter("hydrogen_atom", folder=folder, verbose=verbose, error_verbose=verbose)
    # distances = [0.2, 0.35, 0.45, 1.5, 2.8]
    # distances = np.arange(0.2, 2.90, 0.05)
    # experiments = [(f"h2_r_{R}_linear_growth_ROT-CNOT_2-qubits_100-population_100-optimizer-steps", "[1]", f"R = {R}") for R in distances]
    experiments = [
        # ("h2_all_linear_growth_ROT-CNOT_2-qubits_100-population_200-optimizer-steps", "[1]", "Linear growth"),
        ("gs_h2_errorless_linear_growth_ROT-CNOT_2-qubits_100-population_100-optimizer-steps", "[0]", "Linear growth"),
        ("h2_all_qneat_ROT-CNOT_2-qubits_100-population_200-optimizer-steps", "[9]", "qneat")
    ]
    # print(experiments)
    plotter.add_experiments(experiments)
    # plotter.plot_all(show, save)
    plot_h2_solution(color="r", linewidth=1)
    plot_UCCSD_result_h2(color=UCCSD_COLOR, marker="x")
    plotter.plot_evaluation(show, save, marker = "x")
    plot_UCCSD_diff_h2(color=UCCSD_COLOR, marker="x")
    plotter.plot_delta_evaluation(show, save, marker="x")
    plot_UCCSD_diff_h2(color=UCCSD_COLOR, marker="x")
    plotter.plot_delta_evaluation(show, save, marker="x", logarithmic=True)

def new_results(folder, verbose, show = False, save = False):
    if verbose >= 1:
        print("New results")
    for molecule, n_qubits in [("H2", 2), ("H6",6), ("LiH",8)]:
        plotter = MultipleExperimentPlotter(f"{molecule}_comparisons", folder=folder, verbose=verbose, error_verbose=verbose)
        experiments = [
            # (f"gs_{molecule.lower()}_errorless_saveh_linear_growth_ROT-CNOT_{n_qubits}-qubits_100-population", "[0]", "Linear growth, no optimization"),
            (f"gs_{molecule.lower()}_errorless_saveh_linear_growth_ROT-CNOT_{n_qubits}-qubits_100-population_100-optimizer-steps", "[0]", "Linear growth, 100 steps"),
            # (f"gs_{molecule.lower()}_errorless_saveh_linear_growth_ROT-CNOT_{n_qubits}-qubits_100-population_200-optimizer-steps", "[0]", "Linear growth, 200 steps"),
            # (f"gs_{molecule.lower()}_errorless_saveh_qneat_ROT-CNOT_{n_qubits}-qubits_100-population", "[0]", "Qneat, no optimization"),
            # (f"gs_{molecule.lower()}_errorless_saveh_qneat_ROT-CNOT_{n_qubits}-qubits_100-population_100-optimizer-steps", "[0]", "Qneat, 100 steps"),
            # (f"gs_{molecule.lower()}_errorless_saveh_qneat_ROT-CNOT_{n_qubits}-qubits_100-population_200-optimizer-steps", "[0]", "Qneat, 200 steps"),
        ]
        plotter.add_experiments(experiments)
        plotter.plot_all_generations(show, save)
        gse = GroundStateEnergy(None, molecule.lower())
        gse.plot_solution(color="r", linewidth=1, label="Solution (ED)")
        gse.plot_UCCSD_result(color=UCCSD_COLOR, marker=".")
        gse.plot_adaptVQE_result(color="red", marker=".")
        gse.plot_HE_result(1, color="green", marker=".")
        plotter.plot_evaluation(show, save, marker = "x")
        gse.plot_UCCSD_diff(color=UCCSD_COLOR, marker=".")
        gse.plot_adaptVQE_diff(color="red", marker=".")
        gse.plot_HE_diff(1, color="green", marker=".")
        gse.plot_UCCSD_diff(color=UCCSD_COLOR, marker=".")
        gse.plot_adaptVQE_diff(color="red", marker=".")
        gse.plot_HE_diff(1, color="green", marker=".")

def noise(folder, verbose, show=False, save=False):
    if verbose >= 1:
        print("noise")
    colormap = "cool"
    for molecule, n_qubits in [("H2", 2), ("H6", 6), ("LiH", 8)]:
        for method, method_name in [("linear_growth", "Linear growth"), ("qneat", "Qneat")]:
            for phys_noise, phys_noise_name in [("", ""), ("_phys-noise", ", physical noise")]:
                plotter = MultipleExperimentPlotter(f"{molecule.lower()}_noise_{method}{phys_noise}", folder=folder, verbose=verbose, error_verbose=verbose)
                experiments = []
                for n_shots in range(0, 13):
                    experiments.append((
                        f"gs_{molecule.lower()}_errorless_saveh_{method}_ROT-CNOT_{n_qubits}-qubits_100-population_100-optimizer-steps_{n_shots}-shots{phys_noise}",
                        "[0]",
                        f"{cluster_n_shots[n_shots]}"
                        ))
                plotter.add_experiments(experiments)
                plotter.plot_all_generations(show, save)
                gse = GroundStateEnergy(None, molecule.lower())
                gse.plot_solution(color="r", linewidth=1, label="Solution (ED)")
                gse.plot_UCCSD_result(color=UCCSD_COLOR, marker="x")
                plotter.plot_evaluation(show, save, marker = "x", colormap=colormap)
                gse.plot_UCCSD_diff(color=UCCSD_COLOR, marker="x")
                plotter.plot_delta_evaluation(show, save, marker="x", colormap=colormap)
                gse.plot_UCCSD_diff(color=UCCSD_COLOR, marker="x")
                plotter.plot_delta_evaluation(show, save, marker="x", colormap=colormap, logarithmic=True)
                plotter.plot_box("n_shots", f"{molecule}{phys_noise}", show=show, save=save)
                plotter.plot_box_log("n_shots", f"{molecule}{phys_noise}", show=show, save=save)

def noise_all(folder, verbose, show=False, save=False):
    if verbose >= 1:
        print("noise")
    colormap = "cool"
    for molecule, n_qubits in [("H2", 2), ("H6", 6), ("LiH", 8)]:
        for method, method_name in [("linear_growth", "Linear growth"), ("qneat", "Qneat")]:
            for phys_noise, phys_noise_name in [("", ""), ("_phys-noise", ", physical noise")]:
                plotter = MultipleExperimentPlotter(f"{molecule.lower()}_noise_all_{method}{phys_noise}", folder=folder, verbose=verbose, error_verbose=verbose)
                experiments = []
                for n_shots in range(0, 13):
                    experiments.append((
                        f"gs_{molecule.lower()}_errorless_saveh_{method}_ROT-CNOT_{n_qubits}-qubits_100-population_100-optimizer-steps_{n_shots}-shots{phys_noise}",
                        "*",
                        f"{cluster_n_shots[n_shots]}"
                        ))
                plotter.add_experiments(experiments)
                plotter.plot_all_generations(show, save)
                gse = GroundStateEnergy(None, molecule.lower())
                gse.plot_solution(color="r", linewidth=1, label="Solution (ED)")
                gse.plot_UCCSD_result(color=UCCSD_COLOR, marker="x")
                plotter.plot_evaluation(show, save, marker = "x", colormap=colormap)
                gse.plot_UCCSD_diff(color=UCCSD_COLOR, marker="x")
                plotter.plot_delta_evaluation(show, save, marker="x", colormap=colormap)
                gse.plot_UCCSD_diff(color=UCCSD_COLOR, marker="x")
                plotter.plot_delta_evaluation(show, save, marker="x", colormap=colormap, logarithmic=True)
                plotter.plot_box("n_shots", f"{molecule}{phys_noise_name}", show=show, save=save)
                plotter.plot_box_log("n_shots", f"{molecule}{phys_noise_name}", show=show, save=save)

def noise_new(folder, verbose, show=False, save=False):
    if verbose >= 1:
        print("noise")
    colormap = "cool"
    for molecule, n_qubits in [("H2", 2)]:# [("H2", 2), ("H6", 6), ("LiH", 8)]:
        for method, method_name in [("linear_growth", "Linear growth")]:#, ("qneat", "Qneat")]:
            for phys_noise, phys_noise_name in [("", "")]:#, ("_phys-noise", ", physical noise")]:
                plotter = MultipleExperimentPlotter(f"{molecule.lower()}_noise_new_{method}{phys_noise}", folder=folder, verbose=verbose, error_verbose=verbose)
                experiments = []
                for n_shots in range(0, 13):
                    experiments.append((
                        f"noise_new/gs_{molecule.lower()}_errorless_saveh_{method}_ROT-CNOT_{n_qubits}-qubits_100-population_100-optimizer-steps_{cluster_n_shots[n_shots]}-shots{phys_noise}",
                        # "[14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]",
                        "*",
                        f"{cluster_n_shots[n_shots]}"
                        ))
                plotter.add_experiments(experiments)
                plotter.plot_all_generations(show, save)
                gse = GroundStateEnergy(None, molecule.lower())
                gse.plot_solution(color="r", linewidth=1, label="Solution (ED)")
                gse.plot_UCCSD_result(color=UCCSD_COLOR, marker="x")
                plotter.plot_evaluation(show, save, marker = "x", colormap=colormap)
                gse.plot_UCCSD_diff(color=UCCSD_COLOR, marker="x")
                plotter.plot_delta_evaluation(show, save, marker="x", colormap=colormap)
                gse.plot_UCCSD_diff(color=UCCSD_COLOR, marker="x")
                plotter.plot_delta_evaluation(show, save, marker="x", colormap=colormap, logarithmic=True)
                plotter.plot_box("n_shots", f"{molecule}{phys_noise}", show=show, save=save)
                plotter.plot_box_log("n_shots", f"{molecule}{phys_noise}", show=show, save=save)

def hardware_efficient(folder, verbose, show=False, save=False):
    if verbose >=1:
        print("hardware_efficient")
    colormap = "cool"
    layers = [0, 1, 2, 4, 8, 16]
    n_layers = len(layers)
    colormap = mpl.colormaps.get_cmap(colormap).resampled(n_layers)
    for molecule in ["H2", "H6", "LiH"]:
        gse = GroundStateEnergy(None, molecule.lower())
        gse.plot_solution(color="r", linewidth=1, label="Solution (ED)")
        for ind, l in enumerate(layers):
            gse.plot_HE_result(l, color=colormap(ind/n_layers), marker="x")
        plt.title(f"Hardware efficient anzats for {molecule}")
        plt.grid()
        plt.legend()
        plt.xlabel("Distance between atoms (Angstrom)") #TODO angstrom symbol
        plt.ylabel("Energy (a.u.)")
        if save:
            os.makedirs(f"{folder}/figures/hardware_efficient", exist_ok=True)
            plt.savefig(f"{folder}\\figures\\hardware_efficient\\{molecule}.png")
        if show:
            plt.show()
        plt.close()

        for ind, l in enumerate(layers):
            gse.plot_HE_diff(l, color=colormap(ind/n_layers), marker="x")
        plt.title(f"Hardware efficient anzats for {molecule}")
        plt.grid()
        plt.legend()
        plt.xlabel("Distance between atoms (Angstrom)") #TODO angstrom symbol
        plt.ylabel("Delta energy (a.u.)")
        if save:
            os.makedirs(f"{folder}/figures/hardware_efficient", exist_ok=True)
            plt.savefig(f"{folder}\\figures\\hardware_efficient\\{molecule}_diff.png")
        if show:
            plt.show()
        plt.close()

        plt.yscale("log")
        for ind, l in enumerate(layers):
            gse.plot_HE_diff(l, color=colormap(ind/n_layers), marker="x")
        plt.title(f"Hardware efficient anzats for {molecule}")
        plt.grid()
        plt.legend()
        plt.xlabel("Distance between atoms (Angstrom)") #TODO angstrom symbol
        plt.ylabel("Delta energy (a.u.)")
        if save:
            os.makedirs(f"{folder}/figures/hardware_efficient", exist_ok=True)
            plt.savefig(f"{folder}\\figures\\hardware_efficient\\{molecule}_diff_log.png")
        if show:
            plt.show()
        plt.close()

def hardware_efficient_evaluation_total(folder, verbose, show=False, save=False):
    if verbose >=1:
        print("hardware_efficient")
    colormap = "cool"
    layers = [0, 1, 2, 4, 8, 16]
    n_layers = len(layers)
    colormap = mpl.colormaps.get_cmap(colormap).resampled(n_layers)
    for molecule in ["H2", "H6", "LiH"]:
        gse = GroundStateEnergy(None, molecule.lower())
        gse.plot_solution(color="r", linewidth=1, label="Solution (ED)")
        for ind, l in enumerate(layers):
            gse.plot_HE_result_total(l, color=colormap(ind/n_layers), marker="x")
        plt.title(f"Hardware efficient anzats for {molecule}")
        plt.grid()
        plt.legend()
        plt.xlabel("Distance between atoms (Angstrom)") #TODO angstrom symbol
        plt.ylabel("Energy (a.u.)")
        if save:
            os.makedirs(f"{folder}/figures/hardware_efficient_evaluation-total", exist_ok=True)
            plt.savefig(f"{folder}\\figures\\hardware_efficient_evaluation-total\\{molecule}.png")
        if show:
            plt.show()
        plt.close()

        for ind, l in enumerate(layers):
            gse.plot_HE_diff_total(l, color=colormap(ind/n_layers), marker="x")
        plt.title(f"Hardware efficient anzats for {molecule}")
        plt.grid()
        plt.legend()
        plt.xlabel("Distance between atoms (Angstrom)") #TODO angstrom symbol
        plt.ylabel("Delta energy (a.u.)")
        if save:
            os.makedirs(f"{folder}/figures/hardware_efficient_evaluation-total", exist_ok=True)
            plt.savefig(f"{folder}\\figures\\hardware_efficient_evaluation-total\\{molecule}_diff.png")
        if show:
            plt.show()
        plt.close()

        plt.yscale("log")
        for ind, l in enumerate(layers):
            gse.plot_HE_diff_total(l, color=colormap(ind/n_layers), marker="x")
        plt.title(f"Hardware efficient anzats for {molecule}")
        plt.grid()
        plt.legend()
        plt.xlabel("Distance between atoms (Angstrom)") #TODO angstrom symbol
        plt.ylabel("Delta energy (a.u.)")
        if save:
            os.makedirs(f"{folder}/figures/hardware_efficient_evaluation-total", exist_ok=True)
            plt.savefig(f"{folder}\\figures\\hardware_efficient_evaluation-total\\{molecule}_diff_log.png")
        if show:
            plt.show()
        plt.close()

def hardware_efficient_noise(folder, verbose, show=False, save=False):
    if verbose >= 1:
        print("hardware_efficient_noise")
    colormap = "cool"
    layers = [0, 1, 2, 4, 8, 16]
    n_layers = len(layers)
    colormap = mpl.colormaps.get_cmap(colormap).resampled(n_layers)
    for molecule in ["H2", "H6", "LiH"]:
        gse = GroundStateEnergy(None, molecule.lower())
        for total, func in [("", gse.plot_HE_result), ("-total", gse.plot_HE_result_total)]:
            gse.plot_solution(color="r", linewidth=1, label="Solution (ED)")
            for ind, l in enumerate(layers):
                for phys_noise in [False, True]:
                    marker = "x"
                    if phys_noise:
                        marker = "+"
                    func(l, color=colormap(ind/n_layers), marker=marker, phys_noise=phys_noise)
            plt.title(f"Hardware efficient anzats for {molecule}")
            plt.grid()
            plt.legend()
            plt.xlabel("Distance between atoms (Angstrom)") #TODO angstrom symbol
            plt.ylabel("Energy (a.u.)")
            if save:
                os.makedirs(f"{folder}/figures/hardware_efficient_evaluation{total}", exist_ok=True)
                plt.savefig(f"{folder}\\figures\\hardware_efficient_evaluation{total}\\{molecule}.png")
            if show:
                plt.show()
            plt.close()

            for ind, l in enumerate(layers):
                for phys_noise in [False, True]:
                    marker = "x"
                    if phys_noise:
                        marker = "+"
                    gse.plot_HE_diff_total(l, color=colormap(ind/n_layers), marker=marker, phys_noise=phys_noise)
            plt.title(f"Hardware efficient anzats for {molecule}")
            plt.grid()
            plt.legend()
            plt.xlabel("Distance between atoms (Angstrom)") #TODO angstrom symbol
            plt.ylabel("Delta energy (a.u.)")
            if save:
                os.makedirs(f"{folder}/figures/hardware_efficient_evaluation{total}", exist_ok=True)
                plt.savefig(f"{folder}\\figures\\hardware_efficient_evaluation{total}\\{molecule}_diff.png")
            if show:
                plt.show()
            plt.close()

            plt.yscale("log")
            for ind, l in enumerate(layers):
                for phys_noise in [False, True]:
                    marker = "x"
                    if phys_noise:
                        marker = "+"
                    gse.plot_HE_diff_total(l, color=colormap(ind/n_layers), marker=marker, phys_noise=phys_noise)
            plt.title(f"Hardware efficient anzats for {molecule}")
            plt.grid()
            plt.legend()
            plt.xlabel("Distance between atoms (Angstrom)") #TODO angstrom symbol
            plt.ylabel("Delta energy (a.u.)")
            if save:
                os.makedirs(f"{folder}/figures/hardware_efficient_evaluation{total}", exist_ok=True)
                plt.savefig(f"{folder}\\figures\\hardware_efficient_evaluation{total}\\{molecule}_diff_log.png")
            if show:
                plt.show()
            plt.close()


def noise_total_fitness(folder, verbose, show=False, save=False):
    if verbose >= 1:
        print("noise_total_fitness")
    colormap = None #"cool"
    for molecule, n_qubits, n_layers_HE, n_layers_HE_noise in [("H2", 2, 8, 0), ("H6", 6, 0, 8), ("LiH", 8, 0, 0)]:
        # plotter = MultipleExperimentPlotter(f"gs_{molecule.lower()}_noise_total_fitness", folder=folder, verbose=verbose, error_verbose=verbose)
        plotter = MultipleExperimentPlotter(f"gs_{molecule.lower()}_noise_total", folder=folder, verbose=verbose, error_verbose=verbose)
        experiments = []
        for phys_noise, phys_noise_name in [("", ""), ("_phys-noise", ", physical noise")]:
            # for total_energy, total_energy_name in [("", ""), ("_total-energy", ", total energy")]:
            #     for fitness, fitness_name in [("", ""), ("_shared-fitness", ", shared fitness")]:
                    total_energy, total_energy_name = "_total-energy", ""
                    fitness, fitness_name = "", ""
                    experiments.append((
                        f"0/gs_{molecule.lower()}_errorless_saveh_linear_growth_ROT-CNOT_{n_qubits}-qubits_100-population_100-optimizer-steps{total_energy}{fitness}_0-shots{phys_noise}",
                        "*",
                        f"linear_growth{phys_noise_name}{total_energy_name}{fitness_name}"
                        ))
        plotter.add_experiments(experiments)
        plotter.plot_all_generations(show, save)
        gse = GroundStateEnergy(None, molecule.lower())
        gse.plot_solution(color="r", linewidth=1, label="Solution (ED)")
        gse.plot_UCCSD_result(color=UCCSD_COLOR, marker="x")
        gse.plot_HE_result_total(n_layers_HE, color="olive", marker = "x")
        gse.plot_HE_result_total(n_layers_HE_noise, phys_noise=True, color="green", marker = "x")
        plotter.plot_evaluation(show, save, marker = "x", colormap=colormap)
        # plotter.plot_evaluation(show, save, plot_type="line", colormap=colormap)
        gse.plot_UCCSD_diff(color=UCCSD_COLOR, marker="x")
        gse.plot_HE_diff_total(n_layers_HE, color="olive", marker = "x")
        gse.plot_HE_diff_total(n_layers_HE_noise, phys_noise=True, color="green", marker = "x")
        # plotter.plot_delta_evaluation(show, save, marker = "x", colormap=colormap)
        plotter.plot_delta_evaluation(show, save, plot_type = "line", colormap=colormap)
        gse.plot_UCCSD_diff(color=UCCSD_COLOR, marker="x")
        gse.plot_HE_diff_total(n_layers_HE, color="olive", marker = "x")
        gse.plot_HE_diff_total(n_layers_HE_noise, phys_noise=True, color="green", marker = "x")
        # plotter.plot_delta_evaluation(show, save, marker = "x", colormap=colormap, logarithmic=True)
        plotter.plot_delta_evaluation(show, save, plot_type = "line", colormap=colormap, logarithmic=True)

def thesis_separate(folder, verbose, show=False, save=False):
    if verbose >= 1:
        print("thesis separate")
    colormap = "cool"
    for molecule, n_qubits in [("H2", 2), ("H6", 6), ("LiH", 8)]:
        for method, method_name in [("linear_growth", "Linear growth"), ("qneat", "Qneat")]:
            gate_set = [("ROT-CNOT", "R=RxRyRz"), ("R-CNOT", "Rx,Ry,Rz")]
            if method == "qneat":
                gate_set = [gate_set[0]]
            for gate_set, gate_set_name in gate_set:
                for total_energy, total_energy_name in [("", ""), ("_total-energy", ", total energy")]:
                # for total_energy, total_energy_name in [("_total-energy", ", total energy")]:
                    plotter = MultipleExperimentPlotter(f"thesis-separate_{molecule.lower()}_{gate_set}_{method}{total_energy}", folder=folder, verbose=verbose, error_verbose=verbose)
                    plotter.add_experiment(
                            f"thesis_gs_{molecule.lower()}_errorless_saveh_{method}_{gate_set}_{n_qubits}-qubits_100-population_100-optimizer-steps{total_energy}_0-shots",
                            "*",
                            f"noiseless"
                            )
                    for n_shots in range(12, 0, -1):
                        plotter.add_experiment(
                            f"thesis_gs_{molecule.lower()}_errorless_saveh_{method}_{gate_set}_{n_qubits}-qubits_100-population_100-optimizer-steps{total_energy}_{cluster_n_shots[n_shots]}-shots",
                            "*",
                            f"{cluster_n_shots[n_shots]}"
                            )
                    plotter.plot_box("n_shots", f"{molecule}", show=show, save=save)
                    plotter.plot_box_log("n_shots", f"{molecule}", show=show, save=save)
                    plotter.add_experiment(
                            f"thesis_gs_{molecule.lower()}_errorless_saveh_{method}_{gate_set}_{n_qubits}-qubits_100-population_100-optimizer-steps{total_energy}_0-shots_phys-noise",
                            "*",
                            f"physical noise"
                            )
                    plotter.plot_all_generations(show, save, colormap=colormap)
                    gse = GroundStateEnergy(None, molecule.lower())
                    gse.plot_solution(color="r", linewidth=1, label="Solution (ED)")
                    gse.plot_UCCSD_result(color=UCCSD_COLOR, marker="x")
                    plotter.plot_evaluation(show, save, marker = "x", colormap=colormap)
                    gse.plot_UCCSD_diff(color=UCCSD_COLOR, marker="x")
                    plotter.plot_delta_evaluation(show, save, marker="x", colormap=colormap)
                    # plotter.plot_delta_evaluation(show, save, plot_type = "line", colormap=colormap)
                    gse.plot_UCCSD_diff(absolute=True, color=UCCSD_COLOR, marker="x")
                    # plotter.plot_delta_evaluation(show, save, marker="x", colormap=colormap)
                    plotter.plot_delta_evaluation(show, save, absolute=True, plot_type = "line", colormap=colormap)
                    gse.plot_UCCSD_diff(color=UCCSD_COLOR, marker="x")
                    gse.plot_UCCSD_diff(n_shots=0, color=UCCSD_COLOR, marker="+")
                    gse.plot_UCCSD_diff(n_shots=cluster_n_shots[-1], color=UCCSD_COLOR, marker=".")
                    gse.plot_UCCSD_diff(n_shots=0, phys_noise=True, color=UCCSD_COLOR, marker="*")
                    # plotter.plot_delta_evaluation(show, save, marker="x", colormap=colormap, logarithmic=True)
                    plotter.plot_delta_evaluation(show, save, plot_type = "line", colormap=colormap, logarithmic=True)

def thesis_hf(folder, verbose, show=False, save=False):
    if verbose >= 1:
        print("thesis hf")
    for molecule, n_qubits in [("H2", 2), ("H6", 6), ("LiH", 8)]:
        plotter = MultipleExperimentPlotter(f"thesis-hf_{molecule.lower()}", folder=folder, verbose=verbose, error_verbose=verbose)
        for method, method_name in [("", "|0>"), ("_hf", "fock")]:
            plotter.add_experiment(
                f"thesis_gs_{molecule.lower()}_errorless_saveh{method}_linear_growth_R-CNOT_{n_qubits}-qubits_100-population_100-optimizer-steps_0-shots", 
                "*", 
                method_name
                )
        plotter.plot_all_generations(show, save)
        gse = GroundStateEnergy(None, molecule.lower())
        gse.plot_solution(color="r", linewidth=1, label="Solution (ED)")
        # gse.plot_UCCSD_result(color=UCCSD_COLOR, marker="x")
        plotter.plot_evaluation(show, save, marker = "x")
        # gse.plot_UCCSD_diff(color=UCCSD_COLOR, marker="x")
        plotter.plot_delta_evaluation(show, save, marker="x")
        # plotter.plot_delta_evaluation(show, save, plot_type = "line", colormap=colormap)
        # gse.plot_UCCSD_diff(absolute=True, color=UCCSD_COLOR, marker="x")
        # plotter.plot_delta_evaluation(show, save, marker="x", colormap=colormap)
        plotter.plot_delta_evaluation(show, save, absolute=True, plot_type = "line")
        # # gse.plot_UCCSD_diff(color=UCCSD_COLOR, marker="x")
        # gse.plot_UCCSD_diff(n_shots=0, color=UCCSD_COLOR, marker="+")
        # gse.plot_UCCSD_diff(n_shots=cluster_n_shots[-1], color=UCCSD_COLOR, marker=".")
        # gse.plot_UCCSD_diff(n_shots=0, phys_noise=True, color=UCCSD_COLOR, marker="*")
        plotter.plot_delta_evaluation(show, save, marker="x", logarithmic=True, savename="_scatter")
        plotter.plot_delta_evaluation(show, save, plot_type = "line", logarithmic=True, savename="_line")


def thesis(folder, verbose, show=False, save=False):
    if verbose >= 1:
        print("thesis")
    noiseless, phys_noise, shot_noise = True, True, True
    h2, h6, lih = True, True, True
    _print = False
    
    if noiseless:
        if h2: 
            plotter = MultipleExperimentPlotter("thesis/h2", folder=folder, verbose=verbose, error_verbose=verbose)
            plotter.extra_title = ""
            gse = GroundStateEnergy(None, "h2")
            plotter.add_experiment("thesis_gs_h2_errorless_saveh_linear_growth_R-CNOT_2-qubits_100-population_100-optimizer-steps_0-shots",
                                "*",
                                "QASNEAT R", label_n_runs=False)
            plotter.add_experiment("thesis_gs_h2_errorless_saveh_linear_growth_ROT-CNOT_2-qubits_100-population_100-optimizer-steps_0-shots",
                                "*",
                                "QASNEAT ROT", label_n_runs=False)
            plotter.add_experiment("thesis_gs_h2_errorless_saveh_qneat_ROT-CNOT_2-qubits_100-population_100-optimizer-steps_0-shots",
                                "*",
                                "QNEAT", label_n_runs=False)
            if _print:
                plotter.print_n_runs()
                plotter.print_final_data("best_lengths")
                plotter.print_final_data("best_n_parameters")
            else:
                # plotter.plot_all_generations(show, save)
                plt.hlines(y=[18], xmin=[0], xmax=[100], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                plt.hlines(y=[10], xmin=[0], xmax=[100], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                plotter.plot_vs_generations("best_lengths", "Number of gates of best circuit per generation", "#gates", show, save, savename="_compared")
                plt.hlines(y=[14], xmin=[0], xmax=[100], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                plt.hlines(y=[8], xmin=[0], xmax=[100], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                plotter.plot_vs_generations("best_n_parameters", "Number of parameters of best circuit per generation", "#parameters", show, save, savename="_compared")
                plotter.plot_solution()
                gse.plot_UCCSD_result(n_shots=0, color=UCCSD_COLOR, marker="x", label="UCCSD")
                gse.plot_HE_result(layers=1, n_shots=-1, color=HE_COLOR, marker="x", label="Hardware efficient")
                plotter.plot_evaluation(show, save, marker="x")
                gse.plot_UCCSD_diff(n_shots=0, color=UCCSD_COLOR, marker="x", label="UCCSD")
                gse.plot_HE_diff(layers=1, n_shots=-1, color=HE_COLOR, marker="x", label="Hardware efficient")
                plotter.plot_delta_evaluation(show, save, marker="x")
                gse.plot_UCCSD_diff(n_shots=0, absolute=True, color=UCCSD_COLOR, marker="x", label="UCCSD")
                gse.plot_HE_diff(layers=1, n_shots=-1, absolute=True, color=HE_COLOR, marker="x", label="Hardware efficient")
                plotter.plot_delta_evaluation(show, save, marker="x", logarithmic=True, savename="_scatter")
                gse.plot_UCCSD_diff(n_shots=0, absolute=True, color=UCCSD_COLOR, marker="x", label="UCCSD")
                gse.plot_HE_diff(layers=1, n_shots=-1, absolute=True, color=HE_COLOR, marker="x", label="Hardware efficient")
                plotter.plot_delta_evaluation(show, save, plot_type = "line", logarithmic=True, savename="_line")

        if h6: 
            plotter = MultipleExperimentPlotter("thesis/h6", folder=folder, verbose=verbose, error_verbose=verbose)
            plotter.extra_title = ""
            gse = GroundStateEnergy(None, "h6")
            plotter.add_experiment("thesis_gs_h6_errorless_saveh_linear_growth_R-CNOT_6-qubits_100-population_100-optimizer-steps_0-shots",
                                    "*",
                                    "QASNEAT R", label_n_runs=False)
            plotter.add_experiment("thesis_gs_h6_errorless_saveh_linear_growth_ROT-CNOT_6-qubits_100-population_100-optimizer-steps_0-shots",
                                    "*",
                                    "QASNEAT ROT", label_n_runs=False)
            plotter.add_experiment("thesis_gs_h6_errorless_saveh_qneat_ROT-CNOT_6-qubits_100-population_100-optimizer-steps_0-shots",
                                    "*",
                                    "QNEAT", label_n_runs=False)
            # plotter.plot_all_generations(show, save)
            if _print:
                plotter.print_n_runs()
                plotter.print_final_data("best_lengths")
                plotter.print_final_data("best_n_parameters")
            else:
                from matplotlib.axes import Axes
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=[1, 3])
                ax1:Axes; ax2:Axes
                fig.set_size_inches(8,5)
                ax1.set_ylim(4260, 4270)
                ax2.set_ylim(0, 30)
                ax1.spines.bottom.set_visible(False)
                ax2.spines.top.set_visible(False)
                ax1.xaxis.tick_top()
                ax1.tick_params(labeltop=False, size=0)  # don't put tick labels at the top
                ax2.xaxis.tick_bottom()

                d = .5  # proportion of vertical to horizontal extent of the slanted line
                kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                            linestyle="none", color='k', mec='k', mew=1, clip_on=False)
                ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
                ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

                ax1.hlines(y=[4262], xmin=[0], xmax=[100], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                ax2.hlines(y=[4262], xmin=[0], xmax=[100], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                ax2.hlines(y=[26], xmin=[0], xmax=[100], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                # plt.yscale("log")
                plotter._plot_vs_generations("best_lengths", ax=ax2)
                ax1.set_title("Number of gates of best circuit per generation")
                ax1.grid()
                ax2.grid()
                plt.xlabel("Generation")
                plt.ylabel("#gates")
                ax2.legend(loc="upper left", bbox_to_anchor=(1, 0.5))
                fig.tight_layout(pad=2)
                fig.subplots_adjust(hspace=0.1)
                plotter._show_save_close_plot(savename="multiple_experiments_best_lengths_compared_broken", save=save, show=show)

                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=[1, 3])
                ax1:Axes; ax2:Axes
                fig.set_size_inches(8,5)
                ax1.set_ylim(2760, 2770)
                ax2.set_ylim(0, 30)
                ax1.spines.bottom.set_visible(False)
                ax2.spines.top.set_visible(False)
                ax1.xaxis.tick_top()
                ax1.tick_params(labeltop=False, size=0)  # don't put tick labels at the top
                ax2.xaxis.tick_bottom()

                d = .5  # proportion of vertical to horizontal extent of the slanted line
                kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                            linestyle="none", color='k', mec='k', mew=1, clip_on=False)
                ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
                ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

                ax1.hlines(y=[2764], xmin=[0], xmax=[100], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                ax2.hlines(y=[2764], xmin=[0], xmax=[100], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                ax2.hlines(y=[20], xmin=[0], xmax=[100], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                # plt.yscale("log")
                plotter._plot_vs_generations("best_n_parameters", ax=ax2)
                ax1.set_title("Number of parameters of best circuit per generation")
                ax1.grid()
                ax2.grid()
                plt.xlabel("Generation")
                plt.ylabel("#gates")
                ax2.legend(loc="upper left", bbox_to_anchor=(1, 0.5))
                fig.tight_layout(pad=2)
                fig.subplots_adjust(hspace=0.1)
                plotter._show_save_close_plot(savename="multiple_experiments_best_n_parameters_compared_broken", save=save, show=show)

                # plt.hlines(y=[4262], xmin=[0], xmax=[100], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                plt.hlines(y=[26], xmin=[0], xmax=[100], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                plotter.plot_vs_generations("best_lengths", "Number of gates of best circuit per generation", "#gates", show, save, savename="_compared")

                # plt.hlines(y=[2764], xmin=[0], xmax=[100], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                plt.hlines(y=[20], xmin=[0], xmax=[100], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                # plt.yscale("log")
                plotter.plot_vs_generations("best_n_parameters", "Number of parameters of best circuit per generation", "#parameters", show, save, savename="_compared")
                plotter.plot_solution()
                gse.plot_UCCSD_result(n_shots=0, color=UCCSD_COLOR, marker="x", label="UCCSD")
                gse.plot_HE_result(layers=1, n_shots=-1, color=HE_COLOR, marker="x", label="Hardware efficient")
                plotter.plot_evaluation(show, save, marker="x")
                gse.plot_UCCSD_diff(n_shots=0, color=UCCSD_COLOR, marker="x", label="UCCSD")
                gse.plot_HE_diff(layers=1, n_shots=-1, color=HE_COLOR, marker="x", label="Hardware efficient")
                plotter.plot_delta_evaluation(show, save, marker="x")
                gse.plot_UCCSD_diff(n_shots=0, absolute=True, color=UCCSD_COLOR, marker="x", label="UCCSD")
                gse.plot_HE_diff(layers=1, n_shots=-1, absolute=True, color=HE_COLOR, marker="x", label="Hardware efficient")
                plotter.plot_delta_evaluation(show, save, marker="x", logarithmic=True, savename="_scatter")
                gse.plot_UCCSD_diff(n_shots=0, absolute=True, color=UCCSD_COLOR, marker="x", label="UCCSD")
                gse.plot_HE_diff(layers=1, n_shots=-1, absolute=True, color=HE_COLOR, marker="x", label="Hardware efficient")
                plotter.plot_delta_evaluation(show, save, plot_type = "line", logarithmic=True, savename="_line")

        if lih: 
            plotter = MultipleExperimentPlotter("thesis/lih", folder=folder, verbose=verbose, error_verbose=verbose)
            plotter.extra_title = ""
            gse = GroundStateEnergy(None, "lih")
            plotter.add_experiment("thesis_gs_lih_errorless_saveh_linear_growth_R-CNOT_8-qubits_100-population_100-optimizer-steps_0-shots",
                                    "*",
                                    "QASNEAT R", label_n_runs=False)
            plotter.add_experiment("thesis_gs_lih_errorless_saveh_linear_growth_ROT-CNOT_8-qubits_100-population_100-optimizer-steps_0-shots",
                                    "*",
                                    "QASNEAT ROT", label_n_runs=False)
            plotter.add_experiment("thesis_gs_lih_errorless_saveh_qneat_ROT-CNOT_8-qubits_100-population_100-optimizer-steps_0-shots",
                                    "*",
                                    "QNEAT", label_n_runs=False)
            # plotter.plot_all_generations(show, save)
            if _print:
                plotter.print_n_runs()
                plotter.print_final_data("best_lengths")
                plotter.print_final_data("best_n_parameters")
            else:
                from matplotlib.axes import Axes
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=[1, 4.5])
                ax1:Axes; ax2:Axes
                fig.set_size_inches(8,5)
                ax1.set_ylim(3810, 3820)
                ax2.set_ylim(0, 45)
                ax1.spines.bottom.set_visible(False)
                ax2.spines.top.set_visible(False)
                ax1.xaxis.tick_top()
                ax1.tick_params(labeltop=False, size=0)  # don't put tick labels at the top
                ax2.xaxis.tick_bottom()

                d = .5  # proportion of vertical to horizontal extent of the slanted line
                kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                            linestyle="none", color='k', mec='k', mew=1, clip_on=False)
                ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
                ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

                ax1.hlines(y=[3815], xmin=[0], xmax=[100], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                ax2.hlines(y=[3815], xmin=[0], xmax=[100], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                ax2.hlines(y=[40], xmin=[0], xmax=[100], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                # plt.yscale("log")
                plotter._plot_vs_generations("best_lengths", ax=ax2)
                ax1.set_title("Number of gates of best circuit per generation")
                ax1.grid()
                ax2.grid()
                plt.xlabel("Generation")
                plt.ylabel("#gates")
                ax2.legend(loc="upper left", bbox_to_anchor=(1, 0.5))
                fig.tight_layout(pad=2)
                fig.subplots_adjust(hspace=0.1)
                plotter._show_save_close_plot(savename="multiple_experiments_best_lengths_compared_broken", save=save, show=show)

                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=[1, 4.5])
                ax1:Axes; ax2:Axes
                fig.set_size_inches(8,5)
                ax1.set_ylim(2295, 2305)
                ax2.set_ylim(0, 45)
                ax1.spines.bottom.set_visible(False)
                ax2.spines.top.set_visible(False)
                ax1.xaxis.tick_top()
                ax1.tick_params(labeltop=False, size=0)  # don't put tick labels at the top
                ax2.xaxis.tick_bottom()

                d = .5  # proportion of vertical to horizontal extent of the slanted line
                kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                            linestyle="none", color='k', mec='k', mew=1, clip_on=False)
                ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
                ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

                ax1.hlines(y=[2300], xmin=[0], xmax=[100], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                ax2.hlines(y=[2300], xmin=[0], xmax=[100], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                ax2.hlines(y=[32], xmin=[0], xmax=[100], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                # plt.yscale("log")
                plotter._plot_vs_generations("best_n_parameters", ax=ax2)
                ax1.set_title("Number of parameters of best circuit per generation")
                ax1.grid()
                ax2.grid()
                plt.xlabel("Generation")
                plt.ylabel("#gates")
                ax2.legend(loc="upper left", bbox_to_anchor=(1, 0.5))
                fig.tight_layout(pad=2)
                fig.subplots_adjust(hspace=0.1)
                plotter._show_save_close_plot(savename="multiple_experiments_best_n_parameters_compared_broken", save=save, show=show)

                # plt.hlines(y=[4262], xmin=[0], xmax=[100], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                plt.hlines(y=[40], xmin=[0], xmax=[100], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                plotter.plot_vs_generations("best_lengths", "Number of gates of best circuit per generation", "#gates", show, save, savename="_compared")

                # plt.hlines(y=[2764], xmin=[0], xmax=[100], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                plt.hlines(y=[32], xmin=[0], xmax=[100], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                # plt.yscale("log")
                plotter.plot_vs_generations("best_n_parameters", "Number of parameters of best circuit per generation", "#parameters", show, save, savename="_compared")
                plotter.plot_solution()
                gse.plot_UCCSD_result(n_shots=0, color=UCCSD_COLOR, marker="x", label="UCCSD")
                gse.plot_HE_result(layers=1, n_shots=-1, color=HE_COLOR, marker="x", label="Hardware efficient")
                plotter.plot_evaluation(show, save, marker="x")
                gse.plot_UCCSD_diff(n_shots=0, color=UCCSD_COLOR, marker="x", label="UCCSD")
                gse.plot_HE_diff(layers=1, n_shots=-1, color=HE_COLOR, marker="x", label="Hardware efficient")
                plotter.plot_delta_evaluation(show, save, marker="x")
                gse.plot_UCCSD_diff(n_shots=0, absolute=True, color=UCCSD_COLOR, marker="x", label="UCCSD")
                gse.plot_HE_diff(layers=1, n_shots=-1, absolute=True, color=HE_COLOR, marker="x", label="Hardware efficient")
                plotter.plot_delta_evaluation(show, save, marker="x", logarithmic=True, savename="_scatter")
                gse.plot_UCCSD_diff(n_shots=0, absolute=True, color=UCCSD_COLOR, marker="x", label="UCCSD")
                gse.plot_HE_diff(layers=1, n_shots=-1, absolute=True, color=HE_COLOR, marker="x", label="Hardware efficient")
                plotter.plot_delta_evaluation(show, save, plot_type = "line", logarithmic=True, savename="_line")

    if phys_noise:
        if h2:
            plotter = MultipleExperimentPlotter("thesis/h2_phys-noise", folder=folder, verbose=verbose, error_verbose=verbose)
            plotter.extra_title = ""
            gse = GroundStateEnergy(None, "h2")
            plotter.add_experiment("thesis_gs_h2_errorless_saveh_linear_growth_R-CNOT_2-qubits_100-population_100-optimizer-steps_0-shots_phys-noise",
                                "*",
                                "QASNEAT R", label_n_runs=False)
            plotter.add_experiment("thesis_gs_h2_errorless_saveh_linear_growth_ROT-CNOT_2-qubits_100-population_100-optimizer-steps_0-shots_phys-noise",
                                "*",
                                "QASNEAT ROT", label_n_runs=False)
            plotter.add_experiment("thesis_gs_h2_errorless_saveh_qneat_ROT-CNOT_2-qubits_100-population_100-optimizer-steps_0-shots_phys-noise",
                                "*",
                                "QNEAT", label_n_runs=False)
            # plotter.plot_all_generations(show, save)
            if _print:
                plotter.print_n_runs()
                plotter.print_final_data("best_lengths")
                plotter.print_final_data("best_n_parameters")
            else:
                plt.hlines(y=[18], xmin=[0], xmax=[100], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                plt.hlines(y=[4], xmin=[0], xmax=[100], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                plotter.plot_vs_generations("best_lengths", "Number of gates of best circuit per generation", "#gates", show, save, savename="_compared")
                plt.hlines(y=[14], xmin=[0], xmax=[100], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                plt.hlines(y=[4], xmin=[0], xmax=[100], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                plotter.plot_vs_generations("best_n_parameters", "Number of parameters of best circuit per generation", "#parameters", show, save, savename="_compared")
                plotter.plot_solution()
                gse.plot_UCCSD_result(n_shots=0, color=UCCSD_COLOR, marker="x", label="UCCSD", phys_noise=True)
                gse.plot_HE_result(layers=0, n_shots=-1, color=HE_COLOR, marker="x", label="Hardware efficient", phys_noise=True)
                plotter.plot_evaluation(show, save, marker="x")
                gse.plot_UCCSD_diff(n_shots=0, color=UCCSD_COLOR, marker="x", label="UCCSD", phys_noise=True)
                gse.plot_HE_diff(layers=0, n_shots=-1, color=HE_COLOR, marker="x", label="Hardware efficient", phys_noise=True)
                plotter.plot_delta_evaluation(show, save, marker="x")
                gse.plot_UCCSD_diff(n_shots=0, absolute=True, color=UCCSD_COLOR, marker="x", label="UCCSD", phys_noise=True)
                gse.plot_HE_diff(layers=0, n_shots=-1, absolute=True, color=HE_COLOR, marker="x", label="Hardware efficient", phys_noise=True)
                plotter.plot_delta_evaluation(show, save, marker="x", logarithmic=True, savename="_scatter")
                gse.plot_UCCSD_diff(n_shots=0, absolute=True, color=UCCSD_COLOR, marker="x", label="UCCSD", phys_noise=True)
                gse.plot_HE_diff(layers=0, n_shots=-1, absolute=True, color=HE_COLOR, marker="x", label="Hardware efficient", phys_noise=True)
                plotter.plot_delta_evaluation(show, save, plot_type = "line", logarithmic=True, savename="_line")

        if h6:
            plotter = MultipleExperimentPlotter("thesis/h6_phys-noise", folder=folder, verbose=verbose, error_verbose=verbose)
            plotter.extra_title = ""
            gse = GroundStateEnergy(None, "h6")
            plotter.add_experiment("thesis_gs_h6_errorless_saveh_linear_growth_R-CNOT_6-qubits_100-population_100-optimizer-steps_0-shots_phys-noise",
                                    "*",
                                    "QASNEAT R", label_n_runs=False)
            plotter.add_experiment("thesis_gs_h6_errorless_saveh_linear_growth_ROT-CNOT_6-qubits_100-population_100-optimizer-steps_0-shots_phys-noise",
                                    "*",
                                    "QASNEAT ROT", label_n_runs=False)
            plotter.add_experiment("thesis_gs_h6_errorless_saveh_qneat_ROT-CNOT_6-qubits_100-population_100-optimizer-steps_0-shots_phys-noise",
                                    "*",
                                    "QNEAT", label_n_runs=False)
            if _print:
                plotter.print_n_runs()
                plotter.print_final_data("best_lengths")
                plotter.print_final_data("best_n_parameters")
            else:
                # plotter.plot_all_generations(show, save)
                from matplotlib.axes import Axes
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=[1, 3])
                ax1:Axes; ax2:Axes
                fig.set_size_inches(8,5)
                ax1.set_ylim(4260, 4270)
                ax2.set_ylim(0, 25)
                ax1.spines.bottom.set_visible(False)
                ax2.spines.top.set_visible(False)
                ax1.xaxis.tick_top()
                ax1.tick_params(labeltop=False, size=0)  # don't put tick labels at the top
                ax2.xaxis.tick_bottom()

                d = .5  # proportion of vertical to horizontal extent of the slanted line
                kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                            linestyle="none", color='k', mec='k', mew=1, clip_on=False)
                ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
                ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

                ax1.hlines(y=[4262], xmin=[0], xmax=[100], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                ax2.hlines(y=[4262], xmin=[0], xmax=[100], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                ax2.hlines(y=[12], xmin=[0], xmax=[100], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                # plt.yscale("log")
                plotter._plot_vs_generations("best_lengths", ax=ax2)
                ax1.set_title("Number of gates of best circuit per generation")
                ax1.grid()
                ax2.grid()
                plt.xlabel("Generation")
                plt.ylabel("#gates")
                ax2.legend(loc="upper left", bbox_to_anchor=(1, 0.5))
                fig.tight_layout(pad=2)
                fig.subplots_adjust(hspace=0.1)
                plotter._show_save_close_plot(savename="multiple_experiments_best_lengths_compared_broken", save=save, show=show)

                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=[1, 3])
                ax1:Axes; ax2:Axes
                fig.set_size_inches(8,5)
                ax1.set_ylim(2760, 2770)
                ax2.set_ylim(0, 25)
                ax1.spines.bottom.set_visible(False)
                ax2.spines.top.set_visible(False)
                ax1.xaxis.tick_top()
                ax1.tick_params(labeltop=False, size=0)  # don't put tick labels at the top
                ax2.xaxis.tick_bottom()

                d = .5  # proportion of vertical to horizontal extent of the slanted line
                kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                            linestyle="none", color='k', mec='k', mew=1, clip_on=False)
                ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
                ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

                ax1.hlines(y=[2764], xmin=[0], xmax=[100], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                ax2.hlines(y=[2764], xmin=[0], xmax=[100], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                ax2.hlines(y=[12], xmin=[0], xmax=[100], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                # plt.yscale("log")
                plotter._plot_vs_generations("best_n_parameters", ax=ax2)
                ax1.set_title("Number of parameters of best circuit per generation")
                ax1.grid()
                ax2.grid()
                plt.xlabel("Generation")
                plt.ylabel("#gates")
                ax2.legend(loc="upper left", bbox_to_anchor=(1, 0.5))
                fig.tight_layout(pad=2)
                fig.subplots_adjust(hspace=0.1)
                plotter._show_save_close_plot(savename="multiple_experiments_best_n_parameters_compared_broken", save=save, show=show)

                # plt.hlines(y=[4262], xmin=[0], xmax=[100], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                plt.hlines(y=[12], xmin=[0], xmax=[100], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                plotter.plot_vs_generations("best_lengths", "Number of gates of best circuit per generation", "#gates", show, save, savename="_compared")

                # plt.hlines(y=[2764], xmin=[0], xmax=[100], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                plt.hlines(y=[12], xmin=[0], xmax=[100], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                # plt.yscale("log")
                plotter.plot_vs_generations("best_n_parameters", "Number of parameters of best circuit per generation", "#parameters", show, save, savename="_compared")
                plotter.plot_solution()
                gse.plot_UCCSD_result(n_shots=0, color=UCCSD_COLOR, marker="x", label="UCCSD", phys_noise=True)
                gse.plot_HE_result(layers=0, n_shots=-1, color=HE_COLOR, marker="x", label="Hardware efficient", phys_noise=True)
                plotter.plot_evaluation(show, save, marker="x")
                gse.plot_UCCSD_diff(n_shots=0, color=UCCSD_COLOR, marker="x", label="UCCSD", phys_noise=True)
                gse.plot_HE_diff(layers=0, n_shots=-1, color=HE_COLOR, marker="x", label="Hardware efficient", phys_noise=True)
                plotter.plot_delta_evaluation(show, save, marker="x")
                gse.plot_UCCSD_diff(n_shots=0, absolute=True, color=UCCSD_COLOR, marker="x", label="UCCSD", phys_noise=True)
                gse.plot_HE_diff(layers=0, n_shots=-1, absolute=True, color=HE_COLOR, marker="x", label="Hardware efficient", phys_noise=True)
                plotter.plot_delta_evaluation(show, save, marker="x", logarithmic=True, savename="_scatter")
                gse.plot_UCCSD_diff(n_shots=0, absolute=True, color=UCCSD_COLOR, marker="x", label="UCCSD", phys_noise=True)
                gse.plot_HE_diff(layers=0, n_shots=-1, absolute=True, color=HE_COLOR, marker="x", label="Hardware efficient", phys_noise=True)
                plotter.plot_delta_evaluation(show, save, plot_type = "line", logarithmic=True, savename="_line")

    if shot_noise:
        if h2:
            plotter = MultipleExperimentPlotter(f"thesis/h2_shot-noise", folder=folder, verbose=verbose, error_verbose=verbose)
            plotter.extra_title = ""
            gse = GroundStateEnergy(None, "h2")

            plotter_R = MultipleExperimentPlotter(f"thesis/h2_shot-noise", folder=folder, verbose=verbose, error_verbose=verbose)
            plotter_R.extra_title = ""
            for n_shots in range(1, 13):
                plotter_R.add_experiment(
                    f"thesis_gs_h2_errorless_saveh_linear_growth_R-CNOT_2-qubits_100-population_100-optimizer-steps_{cluster_n_shots[n_shots]}-shots",
                    "*",
                    f"{cluster_n_shots[n_shots]}", label_n_runs=False
                    )
            plotter_R.add_experiment(
                f"thesis_gs_h2_errorless_saveh_linear_growth_R-CNOT_2-qubits_100-population_100-optimizer-steps_0-shots",
                "*",
                f"\u221e", label_n_runs=False
                )
            
            plotter_ROT = MultipleExperimentPlotter(f"thesis/h2_shot-noise", folder=folder, verbose=verbose, error_verbose=verbose)
            plotter_ROT.extra_title = ""
            for n_shots in range(1, 13):
                plotter_ROT.add_experiment(
                    f"thesis_gs_h2_errorless_saveh_linear_growth_ROT-CNOT_2-qubits_100-population_100-optimizer-steps_{cluster_n_shots[n_shots]}-shots",
                    "*",
                    f"{cluster_n_shots[n_shots]}", label_n_runs=False
                    )
            plotter_ROT.add_experiment(
                f"thesis_gs_h2_errorless_saveh_linear_growth_ROT-CNOT_2-qubits_100-population_100-optimizer-steps_0-shots",
                "*",
                f"\u221e", label_n_runs=False
                )

            plotter_qneat = MultipleExperimentPlotter(f"thesis/h2_shot-noise", folder=folder, verbose=verbose, error_verbose=verbose)
            plotter_qneat.extra_title = ""
            for n_shots in range(1, 13):
                plotter_qneat.add_experiment(
                    f"thesis_gs_h2_errorless_saveh_qneat_ROT-CNOT_2-qubits_100-population_100-optimizer-steps_{cluster_n_shots[n_shots]}-shots",
                    "*",
                    f"{cluster_n_shots[n_shots]}", label_n_runs=False
                    )
            plotter_qneat.add_experiment(
                f"thesis_gs_h2_errorless_saveh_qneat_ROT-CNOT_2-qubits_100-population_100-optimizer-steps_0-shots",
                "*",
                f"\u221e", label_n_runs=False
                )
            if _print:
                for _plotter in [plotter_R, plotter_ROT, plotter_qneat]:
                    _plotter.print_n_runs()
                    # _plotter.print_final_data("best_lengths")
                    # _plotter.print_final_data("best_n_parameters")
            else:
                plt.hlines(y=[18], xmin=[0], xmax=[100], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                plt.hlines(y=[10], xmin=[0], xmax=[100], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                plotter_R._plot_shots_generations("best_lengths", label="QASNEAT R")
                plotter_ROT._plot_shots_generations("best_lengths", label="QASNEAT ROT")
                plotter_qneat._plot_shots_generations("best_lengths", label="QNEAT")
                plotter.finalise_plot(
                    title="Number of gates with shot noise",
                    xlabel="#shots",
                    ylabel="#gates",
                    legend=True,
                    savename="shotplot_best_lengths",
                    save=save, show=show,
                )

                plt.hlines(y=[14], xmin=[0], xmax=[100], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                plt.hlines(y=[8], xmin=[0], xmax=[100], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                plotter_R._plot_shots_generations("best_n_parameters", label="QASNEAT R")
                plotter_ROT._plot_shots_generations("best_n_parameters", label="QASNEAT ROT")
                plotter_qneat._plot_shots_generations("best_n_parameters", label="QNEAT")
                plotter.finalise_plot(
                    title="Number of parameters with shot noise",
                    xlabel="#shots",
                    ylabel="#parameters",
                    legend=True,
                    savename="shotplot_best_n_parameters",
                    save=save, show=show,
                )

                plotter_R._plot_shots(label="QASNEAT R")
                plotter_ROT._plot_shots(label="QASNEAT ROT")
                plotter_qneat._plot_shots(label="QNEAT")
                gse.plot_HE_shots(1, label="HE", color=HE_COLOR)
                gse.plot_UCCSD_shots(label="UCCSD", color=UCCSD_COLOR)
                plotter.finalise_plot(
                    title="Energy with shot noise",
                    xlabel="#shots",
                    ylabel="Energy difference",
                    legend=True,
                    savename="shotplot",
                    save=save, show=show,
                )

                plotter_R._plot_shots(label="QASNEAT R", absolute=True)
                plotter_ROT._plot_shots(label="QASNEAT ROT", absolute=True)
                plotter_qneat._plot_shots(label="QNEAT", absolute=True)
                gse.plot_HE_shots(1, label="HE", color=HE_COLOR, absolute=True)
                gse.plot_UCCSD_shots(label="UCCSD", color=UCCSD_COLOR, absolute=True)
                plotter.finalise_plot(
                    title="Energy with shot noise",
                    xlabel="#shots",
                    ylabel="|Energy difference|",
                    legend=True,
                    savename="shotplot_abs",
                    save=save, show=show,
                )

                plt.yscale("log")
                plotter_R._plot_shots(label="QASNEAT R", absolute=True)
                plotter_ROT._plot_shots(label="QASNEAT ROT", absolute=True)
                plotter_qneat._plot_shots(label="QNEAT", absolute=True)
                gse.plot_HE_shots(1, label="HE", color=HE_COLOR, absolute=True)
                gse.plot_UCCSD_shots(label="UCCSD", color=UCCSD_COLOR, absolute=True)
                plotter.finalise_plot(
                    title="Energy with shot noise",
                    xlabel="#shots",
                    ylabel="|Energy difference|",
                    legend=True,
                    savename="shotplot_log",
                    save=save, show=show,
                )

                # eb = ("pi", 90)
                # plt.yscale("log")
                # plotter_R._plot_shots(label="QASNEAT R", absolute=True, errorbar=eb)
                # plotter_ROT._plot_shots(label="QASNEAT ROT", absolute=True, errorbar=eb)
                # plotter_qneat._plot_shots(label="QNEAT", absolute=True, errorbar=eb)
                # gse.plot_HE_shots(1, label="HE", color=HE_COLOR, absolute=True, errorbar=eb)
                # gse.plot_UCCSD_shots(label="UCCSD", color=UCCSD_COLOR, absolute=True, errorbar=eb)
                # plotter.finalise_plot(
                #     title="Energy with shot noise",
                #     xlabel="#shots",
                #     ylabel="|Energy difference|",
                #     legend=True,
                #     savename="shotplot_log_new",
                #     save=save, show=show,
                # )

                plotter_R.plot_box("n_shots", f"QASNEAT R", show=show, save=save, savename="_R")
                plotter_R.plot_box_log("n_shots", f"QASNEAT R", show=show, save=save, savename="_R")
                plotter_ROT.plot_box("n_shots", f"QASNEAT ROT", show=show, save=save, savename="_ROT")
                plotter_ROT.plot_box_log("n_shots", f"QASNEAT ROT", show=show, save=save, savename="_ROT")
                plotter_qneat.plot_box("n_shots", f"QNEAT", show=show, save=save, savename="_qneat")
                plotter_qneat.plot_box_log("n_shots", f"QNEAT", show=show, save=save, savename="_qneat")

        if h6:
            plotter = MultipleExperimentPlotter(f"thesis/h6_shot-noise", folder=folder, verbose=verbose, error_verbose=verbose)
            plotter.extra_title = ""
            gse = GroundStateEnergy(None, "h6")

            plotter_R = MultipleExperimentPlotter(f"thesis/h6_shot-noise", folder=folder, verbose=verbose, error_verbose=verbose)
            plotter_R.extra_title = ""
            for n_shots in range(1, 13):
                plotter_R.add_experiment(
                    f"thesis_gs_h6_errorless_saveh_linear_growth_R-CNOT_6-qubits_100-population_100-optimizer-steps_{cluster_n_shots[n_shots]}-shots",
                    "*",
                    f"{cluster_n_shots[n_shots]}", label_n_runs=False
                    )
            plotter_R.add_experiment(
                f"thesis_gs_h6_errorless_saveh_linear_growth_R-CNOT_6-qubits_100-population_100-optimizer-steps_0-shots",
                "*",
                f"\u221e", label_n_runs=False
                )
            
            plotter_ROT = MultipleExperimentPlotter(f"thesis/h6_shot-noise", folder=folder, verbose=verbose, error_verbose=verbose)
            plotter_ROT.extra_title = ""
            for n_shots in range(1, 13):
                plotter_ROT.add_experiment(
                    f"thesis_gs_h6_errorless_saveh_linear_growth_ROT-CNOT_6-qubits_100-population_100-optimizer-steps_{cluster_n_shots[n_shots]}-shots",
                    "*",
                    f"{cluster_n_shots[n_shots]}", label_n_runs=False
                    )
            plotter_ROT.add_experiment(
                f"thesis_gs_h6_errorless_saveh_linear_growth_ROT-CNOT_6-qubits_100-population_100-optimizer-steps_0-shots",
                "*",
                f"\u221e", label_n_runs=False
                )

            plotter_qneat = MultipleExperimentPlotter(f"thesis/h6_shot-noise", folder=folder, verbose=verbose, error_verbose=verbose)
            plotter_qneat.extra_title = ""
            for n_shots in range(1, 13):
                plotter_qneat.add_experiment(
                    f"thesis_gs_h6_errorless_saveh_qneat_ROT-CNOT_6-qubits_100-population_100-optimizer-steps_{cluster_n_shots[n_shots]}-shots",
                    "*",
                    f"{cluster_n_shots[n_shots]}", label_n_runs=False
                    )
            plotter_qneat.add_experiment(
                f"thesis_gs_h6_errorless_saveh_qneat_ROT-CNOT_6-qubits_100-population_100-optimizer-steps_0-shots",
                "*",
                f"\u221e", label_n_runs=False
                )
            
            if _print:
                for _plotter in [plotter_R, plotter_ROT, plotter_qneat]:
                    _plotter.print_n_runs()
                    # _plotter.print_final_data("best_lengths")
                    # _plotter.print_final_data("best_n_parameters")
            else:
                from matplotlib.axes import Axes
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=[1, 7])
                ax1:Axes; ax2:Axes
                fig.set_size_inches(8,5)
                ax1.set_ylim(4260, 4265)
                ax2.set_ylim(0, 35)
                ax1.spines.bottom.set_visible(False)
                ax2.spines.top.set_visible(False)
                ax1.xaxis.tick_top()
                ax1.tick_params(labeltop=False, size=0)  # don't put tick labels at the top
                ax2.xaxis.tick_bottom()

                d = .5  # proportion of vertical to horizontal extent of the slanted line
                kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                            linestyle="none", color='k', mec='k', mew=1, clip_on=False)
                ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
                ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

                ax1.hlines(y=[4262], xmin=[0], xmax=[len(cluster_n_shots)-1], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                ax2.hlines(y=[4262], xmin=[0], xmax=[len(cluster_n_shots)-1], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                ax2.hlines(y=[26], xmin=[0], xmax=[len(cluster_n_shots)-1], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                # plt.yscale("log")
                plotter_R._plot_shots_generations("best_lengths", label="QASNEAT R", ax=ax2)
                plotter_ROT._plot_shots_generations("best_lengths", label="QASNEAT ROT", ax=ax2)
                plotter_qneat._plot_shots_generations("best_lengths", label="QNEAT", ax=ax2)            
                ax1.set_title("Number of gates with shot noise")
                ax1.grid()
                ax2.grid()
                plt.xlabel("#shots")
                plt.ylabel("#gates")
                ax2.legend(loc="upper left", bbox_to_anchor=(1, 0.5))
                fig.tight_layout(pad=2)
                fig.subplots_adjust(hspace=0.1)
                plotter._show_save_close_plot(savename="shotplot_best_lengths_broken", save=save, show=show)

                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=[1, 7])
                ax1:Axes; ax2:Axes
                fig.set_size_inches(8,5)
                ax1.set_ylim(2760, 2765)
                ax2.set_ylim(0, 35)
                ax1.spines.bottom.set_visible(False)
                ax2.spines.top.set_visible(False)
                ax1.xaxis.tick_top()
                ax1.tick_params(labeltop=False, size=0)  # don't put tick labels at the top
                ax2.xaxis.tick_bottom()

                d = .5  # proportion of vertical to horizontal extent of the slanted line
                kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                            linestyle="none", color='k', mec='k', mew=1, clip_on=False)
                ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
                ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

                ax1.hlines(y=[2764], xmin=[0], xmax=[len(cluster_n_shots)-1], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                ax2.hlines(y=[2764], xmin=[0], xmax=[len(cluster_n_shots)-1], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                ax2.hlines(y=[20], xmin=[0], xmax=[len(cluster_n_shots)-1], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                plotter_R._plot_shots_generations("best_n_parameters", label="QASNEAT R")
                plotter_ROT._plot_shots_generations("best_n_parameters", label="QASNEAT ROT")
                plotter_qneat._plot_shots_generations("best_n_parameters", label="QNEAT")
                # plt.yscale("log")
                ax1.set_title("Number of parameters with shot noise")
                ax1.grid()
                ax2.grid()
                plt.xlabel("#shots")
                plt.ylabel("#gates")
                ax2.legend(loc="upper left", bbox_to_anchor=(1, 0.5))
                fig.tight_layout(pad=2)
                fig.subplots_adjust(hspace=0.1)
                plotter._show_save_close_plot(savename="shotplot_best_n_parameters_broken", save=save, show=show)

                plt.hlines(y=[26], xmin=[0], xmax=[len(cluster_n_shots)-1], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                plotter_R._plot_shots_generations("best_lengths", label="QASNEAT R")
                plotter_ROT._plot_shots_generations("best_lengths", label="QASNEAT ROT")
                plotter_qneat._plot_shots_generations("best_lengths", label="QNEAT")
                plotter.finalise_plot(
                    title="Number of gates with shot noise",
                    xlabel="#shots",
                    ylabel="#gates",
                    legend=True,
                    savename="shotplot_best_lengths",
                    save=save, show=show,
                )

                plt.hlines(y=[20], xmin=[0], xmax=[len(cluster_n_shots)-1], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                plotter_R._plot_shots_generations("best_n_parameters", label="QASNEAT R")
                plotter_ROT._plot_shots_generations("best_n_parameters", label="QASNEAT ROT")
                plotter_qneat._plot_shots_generations("best_n_parameters", label="QNEAT")
                plotter.finalise_plot(
                    title="Number of parameters with shot noise",
                    xlabel="#shots",
                    ylabel="#parameters",
                    legend=True,
                    savename="shotplot_best_n_parameters",
                    save=save, show=show,
                )
                
                plotter_R._plot_shots(label="QASNEAT R")
                plotter_ROT._plot_shots(label="QASNEAT ROT")
                plotter_qneat._plot_shots(label="QNEAT")
                gse.plot_HE_shots(1, label="HE", color=HE_COLOR)
                gse.plot_UCCSD_shots(label="UCCSD", color=UCCSD_COLOR)
                plotter.finalise_plot(
                    title="Energy with shot noise",
                    xlabel="#shots",
                    ylabel="Energy difference",
                    legend=True,
                    savename="shotplot",
                    save=save, show=show,
                )

                plotter_R._plot_shots(label="QASNEAT R", absolute=True)
                plotter_ROT._plot_shots(label="QASNEAT ROT", absolute=True)
                plotter_qneat._plot_shots(label="QNEAT", absolute=True)
                gse.plot_HE_shots(1, label="HE", color=HE_COLOR, absolute=True)
                gse.plot_UCCSD_shots(label="UCCSD", color=UCCSD_COLOR, absolute=True)
                plotter.finalise_plot(
                    title="Energy with shot noise",
                    xlabel="#shots",
                    ylabel="|Energy difference|",
                    legend=True,
                    savename="shotplot_abs",
                    save=save, show=show,
                )

                plt.yscale("log")
                plotter_R._plot_shots(label="QASNEAT R", absolute=True)
                plotter_ROT._plot_shots(label="QASNEAT ROT", absolute=True)
                plotter_qneat._plot_shots(label="QNEAT", absolute=True)
                gse.plot_HE_shots(1, label="HE", color=HE_COLOR, absolute=True)
                gse.plot_UCCSD_shots(label="UCCSD", color=UCCSD_COLOR, absolute=True)
                plotter.finalise_plot(
                    title="Energy with shot noise",
                    xlabel="#shots",
                    ylabel="|Energy difference|",
                    legend=True,
                    savename="shotplot_log",
                    save=save, show=show,
                )

                plotter_R.plot_box("n_shots", f"QASNEAT R", show=show, save=save, savename="_R")
                plotter_R.plot_box_log("n_shots", f"QASNEAT R", show=show, save=save, savename="_R")
                plotter_ROT.plot_box("n_shots", f"QASNEAT ROT", show=show, save=save, savename="_ROT")
                plotter_ROT.plot_box_log("n_shots", f"QASNEAT ROT", show=show, save=save, savename="_ROT")
                plotter_qneat.plot_box("n_shots", f"QNEAT", show=show, save=save, savename="_qneat")
                plotter_qneat.plot_box_log("n_shots", f"QNEAT", show=show, save=save, savename="_qneat")

        if lih:
            plotter = MultipleExperimentPlotter(f"thesis/lih_shot-noise", folder=folder, verbose=verbose, error_verbose=verbose)
            plotter.extra_title = ""
            gse = GroundStateEnergy(None, "lih")

            plotter_R = MultipleExperimentPlotter(f"thesis/lih_shot-noise", folder=folder, verbose=verbose, error_verbose=verbose)
            plotter_R.extra_title = ""
            for n_shots in range(1, 13):
                plotter_R.add_experiment(
                    f"thesis_gs_lih_errorless_saveh_linear_growth_R-CNOT_8-qubits_100-population_100-optimizer-steps_{cluster_n_shots[n_shots]}-shots",
                    "*",
                    f"{cluster_n_shots[n_shots]}", label_n_runs=False
                    )
            plotter_R.add_experiment(
                f"thesis_gs_lih_errorless_saveh_linear_growth_R-CNOT_8-qubits_100-population_100-optimizer-steps_0-shots",
                "*",
                f"\u221e", label_n_runs=False
                )
            
            plotter_ROT = MultipleExperimentPlotter(f"thesis/lih_shot-noise", folder=folder, verbose=verbose, error_verbose=verbose)
            plotter_ROT.extra_title = ""
            for n_shots in range(1, 13):
                plotter_ROT.add_experiment(
                    f"thesis_gs_lih_errorless_saveh_linear_growth_ROT-CNOT_8-qubits_100-population_100-optimizer-steps_{cluster_n_shots[n_shots]}-shots",
                    "*",
                    f"{cluster_n_shots[n_shots]}"
                    )
            plotter_ROT.add_experiment(
                f"thesis_gs_lih_errorless_saveh_linear_growth_ROT-CNOT_8-qubits_100-population_100-optimizer-steps_0-shots",
                "*",
                f"\u221e", label_n_runs=False
                )

            plotter_qneat = MultipleExperimentPlotter(f"thesis/lih_shot-noise", folder=folder, verbose=verbose, error_verbose=verbose)
            plotter_qneat.extra_title = ""
            for n_shots in range(1, 13):
                plotter_qneat.add_experiment(
                    f"thesis_gs_lih_errorless_saveh_qneat_ROT-CNOT_8-qubits_100-population_100-optimizer-steps_{cluster_n_shots[n_shots]}-shots",
                    "*",
                    f"{cluster_n_shots[n_shots]}", label_n_runs=False
                    )
            plotter_qneat.add_experiment(
                f"thesis_gs_lih_errorless_saveh_qneat_ROT-CNOT_8-qubits_100-population_100-optimizer-steps_0-shots",
                "*",
                f"\u221e", label_n_runs=False
                )
            
            if _print:
                for _plotter in [plotter_R, plotter_ROT, plotter_qneat]:
                    _plotter.print_n_runs()
                    # _plotter.print_final_data("best_lengths")
                    # _plotter.print_final_data("best_n_parameters")
            else:
                from matplotlib.axes import Axes
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=[1, 4.5])
                ax1:Axes; ax2:Axes
                fig.set_size_inches(8,5)
                ax1.set_ylim(3810, 3820)
                ax2.set_ylim(0, 45)
                ax1.spines.bottom.set_visible(False)
                ax2.spines.top.set_visible(False)
                ax1.xaxis.tick_top()
                ax1.tick_params(labeltop=False, size=0)  # don't put tick labels at the top
                ax2.xaxis.tick_bottom()

                d = .5  # proportion of vertical to horizontal extent of the slanted line
                kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                            linestyle="none", color='k', mec='k', mew=1, clip_on=False)
                ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
                ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

                ax1.hlines(y=[3815], xmin=[0], xmax=[len(cluster_n_shots)-1], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                ax2.hlines(y=[3815], xmin=[0], xmax=[len(cluster_n_shots)-1], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                ax2.hlines(y=[40], xmin=[0], xmax=[len(cluster_n_shots)-1], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                # plt.yscale("log")
                plotter_R._plot_shots_generations("best_lengths", label="QASNEAT R", ax=ax2)
                plotter_ROT._plot_shots_generations("best_lengths", label="QASNEAT ROT", ax=ax2)
                plotter_qneat._plot_shots_generations("best_lengths", label="QNEAT", ax=ax2)            
                ax1.set_title("Number of gates with shot noise")
                ax1.grid()
                ax2.grid()
                plt.xlabel("#shots")
                plt.ylabel("#gates")
                ax2.legend(loc="upper left", bbox_to_anchor=(1, 0.5))
                fig.tight_layout(pad=2)
                fig.subplots_adjust(hspace=0.1)
                plotter._show_save_close_plot(savename="shotplot_best_lengths_broken", save=save, show=show)

                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=[1, 4.5])
                ax1:Axes; ax2:Axes
                fig.set_size_inches(8,5)
                ax1.set_ylim(2295, 2305)
                ax2.set_ylim(0, 45)
                ax1.spines.bottom.set_visible(False)
                ax2.spines.top.set_visible(False)
                ax1.xaxis.tick_top()
                ax1.tick_params(labeltop=False, size=0)  # don't put tick labels at the top
                ax2.xaxis.tick_bottom()

                d = .5  # proportion of vertical to horizontal extent of the slanted line
                kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                            linestyle="none", color='k', mec='k', mew=1, clip_on=False)
                ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
                ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

                ax1.hlines(y=[2300], xmin=[0], xmax=[len(cluster_n_shots)-1], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                ax2.hlines(y=[2300], xmin=[0], xmax=[len(cluster_n_shots)-1], colors=[UCCSD_COLOR], linestyles="dashed", label="UCCSD")
                ax2.hlines(y=[32], xmin=[0], xmax=[len(cluster_n_shots)-1], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                plotter_R._plot_shots_generations("best_n_parameters", label="QASNEAT R")
                plotter_ROT._plot_shots_generations("best_n_parameters", label="QASNEAT ROT")
                plotter_qneat._plot_shots_generations("best_n_parameters", label="QNEAT")
                # plt.yscale("log")
                ax1.set_title("Number of parameters with shot noise")
                ax1.grid()
                ax2.grid()
                plt.xlabel("#shots")
                plt.ylabel("#gates")
                ax2.legend(loc="upper left", bbox_to_anchor=(1, 0.5))
                fig.tight_layout(pad=2)
                fig.subplots_adjust(hspace=0.1)
                plotter._show_save_close_plot(savename="shotplot_best_n_parameters_broken", save=save, show=show)

                plt.hlines(y=[40], xmin=[0], xmax=[len(cluster_n_shots)-1], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                plotter_R._plot_shots_generations("best_lengths", label="QASNEAT R")
                plotter_ROT._plot_shots_generations("best_lengths", label="QASNEAT ROT")
                plotter_qneat._plot_shots_generations("best_lengths", label="QNEAT")
                plotter.finalise_plot(
                    title="Number of gates with shot noise",
                    xlabel="#shots",
                    ylabel="#gates",
                    legend=True,
                    savename="shotplot_best_lengths",
                    save=save, show=show,
                )

                plt.hlines(y=[32], xmin=[0], xmax=[len(cluster_n_shots)-1], colors=[HE_COLOR], linestyles="dashed", label="Hardware efficient")
                plotter_R._plot_shots_generations("best_n_parameters", label="QASNEAT R")
                plotter_ROT._plot_shots_generations("best_n_parameters", label="QASNEAT ROT")
                plotter_qneat._plot_shots_generations("best_n_parameters", label="QNEAT")
                plotter.finalise_plot(
                    title="Number of parameters with shot noise",
                    xlabel="#shots",
                    ylabel="#parameters",
                    legend=True,
                    savename="shotplot_best_n_parameters",
                    save=save, show=show,
                )

                plotter_R._plot_shots(label="QASNEAT R")
                plotter_ROT._plot_shots(label="QASNEAT ROT")
                plotter_qneat._plot_shots(label="QNEAT")
                gse.plot_HE_shots(1, label="HE", color=HE_COLOR)
                gse.plot_UCCSD_shots(label="UCCSD", color=UCCSD_COLOR)
                plotter.finalise_plot(
                    title="Energy with shot noise",
                    xlabel="#shots",
                    ylabel="Energy difference",
                    legend=True,
                    savename="shotplot",
                    save=save, show=show,
                )

                plotter_R._plot_shots(label="QASNEAT R", absolute=True)
                plotter_ROT._plot_shots(label="QASNEAT ROT", absolute=True)
                plotter_qneat._plot_shots(label="QNEAT", absolute=True)
                gse.plot_HE_shots(1, label="HE", color=HE_COLOR, absolute=True)
                gse.plot_UCCSD_shots(label="UCCSD", color=UCCSD_COLOR, absolute=True)
                plotter.finalise_plot(
                    title="Energy with shot noise",
                    xlabel="#shots",
                    ylabel="|Energy difference|",
                    legend=True,
                    savename="shotplot_abs",
                    save=save, show=show,
                )

                plt.yscale("log")
                plotter_R._plot_shots(label="QASNEAT R", absolute=True)
                plotter_ROT._plot_shots(label="QASNEAT ROT", absolute=True)
                plotter_qneat._plot_shots(label="QNEAT", absolute=True)
                gse.plot_HE_shots(1, label="HE", color=HE_COLOR, absolute=True)
                gse.plot_UCCSD_shots(label="UCCSD", color=UCCSD_COLOR, absolute=True)
                plotter.finalise_plot(
                    title="Energy with shot noise",
                    xlabel="#shots",
                    ylabel="|Energy difference|",
                    legend=True,
                    savename="shotplot_log",
                    save=save, show=show,
                )

                plotter_R.plot_box("n_shots", f"QASNEAT R", show=show, save=save, savename="_R")
                plotter_R.plot_box_log("n_shots", f"QASNEAT R", show=show, save=save, savename="_R")
                plotter_ROT.plot_box("n_shots", f"QASNEAT ROT", show=show, save=save, savename="_ROT")
                plotter_ROT.plot_box_log("n_shots", f"QASNEAT ROT", show=show, save=save, savename="_ROT")
                plotter_qneat.plot_box("n_shots", f"QNEAT", show=show, save=save, savename="_qneat")
                plotter_qneat.plot_box_log("n_shots", f"QNEAT", show=show, save=save, savename="_qneat")


def UCCSD(folder, verbose, show=False, save=False):
    # from qiskit import transpile
    # from qiskit.circuit import Parameter
    # from qiskit.circuit.library import EfficientSU2, RXGate, RYGate, RZGate, CXGate
    # from qiskit.transpiler import Target
    # from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
    # from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper

    # for molecule, n_qubits, orbitals, particles in [("H2", 2, 2, (1,1)), ("H6", 6, 4, (2,2)), ("LiH", 8, 5, (1, 1))]:
    #     print(f"{molecule = }")
    #     print("Parity")
    #     print(HartreeFock(orbitals, particles, ParityMapper(particles)).draw())
    #     print("Jordan Wigner")
    #     print(HartreeFock(orbitals, particles, JordanWignerMapper()).draw())
    #     print("UCCSD ansatz")
    #     target = Target(num_qubits=n_qubits)
    #     target.add_instruction(CXGate(), {(i,(i+1)%n_qubits):None for i in range(n_qubits)})
    #     target.add_instruction(RXGate(Parameter('theta')), {(i,):None for i in range(n_qubits)})
    #     target.add_instruction(RYGate(Parameter('theta')), {(i,):None for i in range(n_qubits)})
    #     target.add_instruction(RZGate(Parameter('theta')), {(i,):None for i in range(n_qubits)})
    #     ansatz = UCCSD(
    #         orbitals,
    #         particles,
    #         ParityMapper(particles),
    #     )
    #     ansatz = transpile(ansatz, target=target)
    #     print(ansatz)

    if verbose >= 1:
        print("UCCSD_noise")
    colormap = "cool"
    colormap = mpl.colormaps.get_cmap(colormap).resampled(len(cluster_n_shots))
    for molecule in ["H2"]:#, "H6", "LiH"]:
        gse = GroundStateEnergy(None, molecule.lower())
        for total, func in [("", gse.plot_UCCSD_result)]:
            gse.plot_solution(color="r", linewidth=1, label="Solution (ED)")
            for ind, l in enumerate(cluster_n_shots):
                for phys_noise in [False, True]:
                    marker = "x"
                    if phys_noise:
                        marker = "+"
                    func(n_shots=l, color=colormap(ind/len(cluster_n_shots)), marker=marker, phys_noise=phys_noise)
            plt.title(f"UCCSD anzats for {molecule}")
            plt.grid()
            plt.legend()
            plt.xlabel("Distance between atoms (Angstrom)") #TODO angstrom symbol
            plt.ylabel("Energy (a.u.)")
            if save:
                os.makedirs(f"{folder}/figures/UCCSD_evaluation{total}", exist_ok=True)
                plt.savefig(f"{folder}\\figures\\UCCSD_evaluation{total}\\{molecule}.png")
            if show:
                plt.show()
            plt.close()

            for ind, l in enumerate(cluster_n_shots):
                for phys_noise in [False, True]:
                    marker = "x"
                    if phys_noise:
                        marker = "+"
                    gse.plot_UCCSD_diff(n_shots=l, color=colormap(ind/len(cluster_n_shots)), marker=marker, phys_noise=phys_noise)
            plt.title(f"UCCSD anzats for {molecule}")
            plt.grid()
            plt.legend()
            plt.xlabel("Distance between atoms (Angstrom)") #TODO angstrom symbol
            plt.ylabel("Delta energy (a.u.)")
            if save:
                os.makedirs(f"{folder}/figures/UCCSD_evaluation{total}", exist_ok=True)
                plt.savefig(f"{folder}\\figures\\UCCSD_evaluation{total}\\{molecule}_diff.png")
            if show:
                plt.show()
            plt.close()

            plt.yscale("log")
            for ind, l in enumerate(cluster_n_shots):
                for phys_noise in [False, True]:
                    marker = "x"
                    if phys_noise:
                        marker = "+"
                    gse.plot_UCCSD_diff(n_shots=l, color=colormap(ind/len(cluster_n_shots)), marker=marker, phys_noise=phys_noise)
            plt.title(f"UCCSD anzats for {molecule}")
            plt.grid()
            plt.legend()
            plt.xlabel("Distance between atoms (Angstrom)") #TODO angstrom symbol
            plt.ylabel("Delta energy (a.u.)")
            if save:
                os.makedirs(f"{folder}/figures/UCCSD_evaluation{total}", exist_ok=True)
                plt.savefig(f"{folder}\\figures\\UCCSD_evaluation{total}\\{molecule}_diff_log.png")
            if show:
                plt.show()
            plt.close()

def test(folder, verbose, show=False, save=False):
    if verbose >= 1:
        print("test")
    colormap = "cool"#"plasma"
    for molecule, n_qubits in [("H2", 2), ("H6", 6)]:
        for method, method_name in [("linear_growth", "Linear growth"), ("qneat", "Qneat")]:
            for phys_noise, phys_noise_name in [("", ""), ("_phys-noise", ", physical noise")]:
                plotter = MultipleExperimentPlotter(f"{molecule.lower()}_noise_all_{method}{phys_noise}", folder=folder, verbose=verbose, error_verbose=verbose)
                experiments = []
                for n_shots in range(0, 13):
                    experiments.append((
                        f"gs_{molecule.lower()}_errorless_saveh_{method}_ROT-CNOT_{n_qubits}-qubits_100-population_100-optimizer-steps_{n_shots}-shots{phys_noise}",
                        "[0]",
                        f"{cluster_n_shots[n_shots]}"
                        ))
                plotter.add_experiments(experiments)
                # plotter.plot_all(show, save)
                gse = GroundStateEnergy(None, molecule.lower())
                # gse.plot_solution(color="r", linewidth=1, label="Solution (ED)")
                # gse.plot_UCCSD_result(color=UCCSD_COLOR, marker="x")
                # plotter.plot_evaluation(show, save, marker = "x")
                gse.plot_UCCSD_diff(color=UCCSD_COLOR, marker="x")
                plotter.plot_delta_evaluation(show, save, marker="x", colormap=colormap)
                gse.plot_UCCSD_diff(color=UCCSD_COLOR, marker="x")
                plotter.plot_delta_evaluation(show, save, marker="x", colormap=colormap, logarithmic=True)
                # plotter.plot_box("n_shots", f"{molecule}{phys_noise_name}", show=show, save=save)
                # plotter.plot_box_log("n_shots", f"{molecule}{phys_noise_name}", show=show, save=save)
            exit()
    
if __name__ == "__main__": 
    from argparse import ArgumentParser
    argparser = ArgumentParser()
    argparser.add_argument("experiment", type=str, help = "Which experiment to plot.")
    argparser.add_argument("--cluster", dest='folder', nargs='?', default=".", const=".\\cluster")
    argparser.add_argument('--verbose', '-v', action='count', default=0)
    argparser.add_argument('--show', action="store_true")
    argparser.add_argument('--save', action="store_true")
    args = argparser.parse_args()
    
    if args.experiment == "constant_population_size" or args.experiment == "all":
        constant_population_size(args.folder, args.verbose, show=args.show, save=args.save)
    if args.experiment == "stagnant_experiment" or args.experiment == "all":
        stagnant_experiment(args.folder, args.verbose, show=args.show, save=args.save)
    if args.experiment == "optimizer_steps" or args.experiment == "all":
        optimizer_steps(args.folder, args.verbose, show=args.show, save=args.save)
    if args.experiment == "hydrogen_atom" or args.experiment == "all":
        hydrogen_atom(args.folder, args.verbose, show=args.show, save=args.save)
    if args.experiment == "new" or args.experiment == "all":
        new_results(args.folder, args.verbose, args.show, args.save)
    if args.experiment == "noise" or args.experiment == "all":
        noise(args.folder, args.verbose, args.show, args.save)
    if args.experiment == "noise_all" or args.experiment == "all":
        noise_all(args.folder, args.verbose, args.show, args.save)
    if args.experiment == "noise_new" or args.experiment == "all":
        noise_new(args.folder, args.verbose, args.show, args.save)
    if args.experiment == "hardware_efficient" or args.experiment == "all":
        hardware_efficient(args.folder, args.verbose, args.show, args.save)
    if args.experiment == "hardware_efficient_evaluation_total" or args.experiment == "all":
        hardware_efficient_evaluation_total(args.folder, args.verbose, args.show, args.save)
    if args.experiment == "hardware_efficient_noise" or args.experiment == "all":
        hardware_efficient_noise(args.folder, args.verbose, args.show, args.save)
    if args.experiment == "noise_total_fitness" or args.experiment == "all":
        noise_total_fitness(args.folder, args.verbose, args.show, args.save)
    if args.experiment == "thesis_separate" or args.experiment == "all":
        thesis_separate(args.folder, args.verbose, args.show, args.save)
    if args.experiment == "thesis_hf" or args.experiment == "all":
        thesis_hf(args.folder, args.verbose, args.show, args.save)
    if args.experiment == "thesis" or args.experiment == "all":
        thesis(args.folder, args.verbose, args.show, args.save)
    if args.experiment == "UCCSD" or args.experiment == "all":
        UCCSD(args.folder, args.verbose, args.show, args.save)
    if args.experiment == "test":
        test(args.folder, args.verbose, args.show, args.save)
