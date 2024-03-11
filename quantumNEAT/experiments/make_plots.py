import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from experiments.plotter import MultipleExperimentPlotter
from quantumneat.problems.chemistry import GroundStateEnergy
from quantumneat.problems.hydrogen import get_solutions, plot_solution as plot_h2_solution, plot_UCCSD_diff as plot_UCCSD_diff_h2, plot_UCCSD_result as plot_UCCSD_result_h2
from experiments.run_experiment import cluster_n_shots

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
    plot_UCCSD_result_h2(color="black", marker="x")
    plotter.plot_evaluation("Evaluation", show, save, marker = "x")
    plot_UCCSD_diff_h2(color="black", marker="x")
    plotter.plot_delta_evaluation(show, save, marker="x")
    plot_UCCSD_diff_h2(color="black", marker="x")
    plotter.plot_delta_evaluation(show, save, marker="x", logarithmic=True)

def new_results(folder, verbose, show = False, save = False):
    if verbose >= 1:
        print("New results")
    for molecule, n_qubits in [("H2", 2), ("H6",6), ("LiH",8)]:
        plotter = MultipleExperimentPlotter(f"{molecule}_comparisons", folder=folder, verbose=verbose, error_verbose=verbose)
        experiments = [
            (f"gs_{molecule.lower()}_errorless_saveh_linear_growth_ROT-CNOT_{n_qubits}-qubits_100-population", "[0]", "Linear growth, no optimization"),
            (f"gs_{molecule.lower()}_errorless_saveh_linear_growth_ROT-CNOT_{n_qubits}-qubits_100-population_100-optimizer-steps", "[0]", "Linear growth, 100 steps"),
            (f"gs_{molecule.lower()}_errorless_saveh_linear_growth_ROT-CNOT_{n_qubits}-qubits_100-population_200-optimizer-steps", "[0]", "Linear growth, 200 steps"),
            (f"gs_{molecule.lower()}_errorless_saveh_qneat_ROT-CNOT_{n_qubits}-qubits_100-population", "[0]", "Qneat, no optimization"),
            (f"gs_{molecule.lower()}_errorless_saveh_qneat_ROT-CNOT_{n_qubits}-qubits_100-population_100-optimizer-steps", "[0]", "Qneat, 100 steps"),
            (f"gs_{molecule.lower()}_errorless_saveh_qneat_ROT-CNOT_{n_qubits}-qubits_100-population_200-optimizer-steps", "[0]", "Qneat, 200 steps"),
        ]
        plotter.add_experiments(experiments)
        plotter.plot_all_generations(show, save)
        gse = GroundStateEnergy(None, molecule.lower())
        gse.plot_solution(color="r", linewidth=1, label="Solution (ED)")
        gse.plot_UCCSD_result(color="black", marker="x")
        plotter.plot_evaluation(f"{molecule} evaluation", show, save, marker = "x")
        gse.plot_UCCSD_diff(color="black", marker="x")
        plotter.plot_delta_evaluation(show, save, marker="x")
        gse.plot_UCCSD_diff(color="black", marker="x")
        plotter.plot_delta_evaluation(show, save, logarithmic = True, marker="x")

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
                gse.plot_UCCSD_result(color="black", marker="x")
                plotter.plot_evaluation(f"{molecule} evaluation", show, save, marker = "x", colormap=colormap)
                gse.plot_UCCSD_diff(color="black", marker="x")
                plotter.plot_delta_evaluation(show, save, marker="x", colormap=colormap)
                gse.plot_UCCSD_diff(color="black", marker="x")
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
                gse.plot_UCCSD_result(color="black", marker="x")
                plotter.plot_evaluation(f"{molecule} evaluation", show, save, marker = "x", colormap=colormap)
                gse.plot_UCCSD_diff(color="black", marker="x")
                plotter.plot_delta_evaluation(show, save, marker="x", colormap=colormap)
                gse.plot_UCCSD_diff(color="black", marker="x")
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
                gse.plot_UCCSD_result(color="black", marker="x")
                plotter.plot_evaluation(f"{molecule} evaluation", show, save, marker = "x", colormap=colormap)
                gse.plot_UCCSD_diff(color="black", marker="x")
                plotter.plot_delta_evaluation(show, save, marker="x", colormap=colormap)
                gse.plot_UCCSD_diff(color="black", marker="x")
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

def test(folder, verbose, show=False, save=False):
    if verbose >= 1:
        print("noise")
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
                # gse.plot_UCCSD_result(color="black", marker="x")
                # plotter.plot_evaluation(f"{molecule} evaluation", show, save, marker = "x")
                gse.plot_UCCSD_diff(color="black", marker="x")
                plotter.plot_delta_evaluation(show, save, marker="x", colormap=colormap)
                gse.plot_UCCSD_diff(color="black", marker="x")
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
    if args.experiment == "test":
        test(args.folder, args.verbose, args.show, args.save)
