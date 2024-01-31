from experiments.plotter import MultipleExperimentPlotter

def constant_population_size():
    plotter = MultipleExperimentPlotter("constant_population_size")
    experiments = [
        ("tfim_linear_growth_ROT-CNOT_5-qubits_100-population_100-optimizer-steps_no-forced-population", "original"),
        ("tfim_linear_growth_ROT-CNOT_5-qubits_100-population_100-optimizer-steps_no-forced-population_normalised-fitness", "normalised-fitness"),
        ("tfim_linear_growth_ROT-CNOT_5-qubits_100-population_100-optimizer-steps", "forced_population"),
        ("tfim_linear_growth_ROT-CNOT_5-qubits_100-population_100-optimizer-steps_normalised-fitness", "forced_population_normalised-fitness"),
    ]
    plotter.add_experiments(experiments, runs="*")
    plotter.plot_all(False, True)

if __name__ == "__main__":
    constant_population_size()
