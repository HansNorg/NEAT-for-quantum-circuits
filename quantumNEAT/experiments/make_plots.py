import numpy as np

from experiments.plotter import MultipleExperimentPlotter
from quantumneat.problems.hydrogen import plot_solution as plot_h2_solution

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

def hydrogen_atom(folder, verbose, show = False, save = False):
    if verbose >= 1:
        print("Hydrogen atom")
    plotter = MultipleExperimentPlotter("hydrogen_atom", folder=folder, verbose=verbose)
    # distances = [0.2, 0.35, 0.45, 1.5, 2.8]
    # experiments = [(f"h2_r_{R}_linear_growth_ROT-CNOT_2-qubits_100-population_100-optimizer-steps", "[1]", f"R = {R}") for R in distances]
    distances = np.arange(0.2, 2.90, 0.05)
    experiments = [(f"h2_r_{R:.2f}_linear_growth_ROT-CNOT_2-qubits_100-population_100-optimizer-steps", "[0]", f"R = {R:.2f}") for R in distances]
    # print(experiments)
    plotter.add_experiments(experiments)
    plotter.plot_all(show, save)
    plot_h2_solution(color="r", linewidth=1)
    plotter.plot_min_energy(distances, "Hydrogen atom", show, save, marker="x")
    # plotter.plot_min_energy(distances, "Hydrogen atom", show, save, marker="x", zorder=2)

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
