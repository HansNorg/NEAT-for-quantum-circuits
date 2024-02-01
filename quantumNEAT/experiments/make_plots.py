from experiments.plotter import MultipleExperimentPlotter

def constant_population_size(folder, verbose):
    plotter = MultipleExperimentPlotter("constant_population_size", folder=folder, verbose=verbose)
    experiments = [
        ("tfim_linear_growth_ROT-CNOT_5-qubits_100-population_100-optimizer-steps_no-forced-population", "*", "original"),
        ("tfim_linear_growth_ROT-CNOT_5-qubits_100-population_100-optimizer-steps_no-forced-population_normalised-fitness", "*", "normalised-fitness"),
        ("tfim_linear_growth_ROT-CNOT_5-qubits_100-population_100-optimizer-steps", 'range(23, 33)', "forced_population"),
        ("tfim_linear_growth_ROT-CNOT_5-qubits_100-population_100-optimizer-steps_normalised-fitness", "*", "forced_population_normalised-fitness"),
    ]
    plotter.add_experiments(experiments)
    plotter.plot_all(False, True)

if __name__ == "__main__":
    from argparse import ArgumentParser
    argparser = ArgumentParser()
    argparser.add_argument("--cluster", dest='folder', nargs='?', default=".", const=".\\cluster")
    argparser.add_argument('--verbose', '-v', action='count', default=0)
    args = argparser.parse_args()
    constant_population_size(args.folder, args.verbose)
