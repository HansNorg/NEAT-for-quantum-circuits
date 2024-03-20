from argparse import ArgumentParser, Namespace
import itertools
from experiments.run_experiment import main, cluster_n_shots

EXPERIMENTS = (
    #(problem, implementation, n_qubits, n_runs, gate_set, phys_noise, n_shots),
    # (["cim"], ["linear_growth", "qneat"], [5], [10], ["ROT-CNOT", "R-CNOT"], [True, False], cluster_n_shots),
    # (["tfim"], ["linear_growth", "qneat"], [5], [10], ["ROT-CNOT", "R-CNOT"], [True, False], cluster_n_shots),
    (["gs_h2_errorless_saveh"], ["linear_growth"], [2], [10], ["ROT-CNOT", "R-CNOT"], [False], cluster_n_shots),
    (["gs_h2_errorless_saveh"], ["linear_growth"], [2], [10], ["ROT-CNOT", "R-CNOT"], [True], [0]),
    (["gs_h2_errorless_saveh"], ["qneat"], [2], [10], ["ROT-CNOT"], [False], cluster_n_shots),
    (["gs_h2_errorless_saveh"], ["qneat"], [2], [10], ["ROT-CNOT"], [True], [0]),
    (["gs_h6_errorless_saveh"], ["linear_growth"], [6], [10], ["ROT-CNOT", "R-CNOT"], [False], cluster_n_shots),
    (["gs_h6_errorless_saveh"], ["linear_growth"], [6], [10], ["ROT-CNOT", "R-CNOT"], [True], [0]),
    (["gs_h6_errorless_saveh"], ["qneat"], [6], [10], ["ROT-CNOT"], [False], cluster_n_shots),
    (["gs_h6_errorless_saveh"], ["qneat"], [6], [10], ["ROT-CNOT"], [True], [0]),
    (["gs_lih_errorless_saveh"], ["linear_growth"], [8], [10], ["ROT-CNOT", "R-CNOT"], [False], cluster_n_shots),
    (["gs_lih_errorless_saveh"], ["linear_growth"], [8], [10], ["ROT-CNOT", "R-CNOT"], [True], [0]),
    (["gs_lih_errorless_saveh"], ["qneat"], [8], [10], ["ROT-CNOT"], [False], cluster_n_shots),
    (["gs_lih_errorless_saveh"], ["qneat"], [8], [10], ["ROT-CNOT"], [True], [0]),
)

def setup_experiment(problem_ids:list[int], experiment_id:int):
    args = Namespace()
    args.population_size = 100
    args.generations = 100
    args.optimizer_steps = 100
    args.number_of_cpus = -1
    args.plot = False
    args.simulator = "qulacs"
    args.total_energy = True
    args.fitness_sharing = False
    args.name = "thesis_"
    args.extra_info = ""

    experiments = []
    for problem_id in problem_ids:
        problem = EXPERIMENTS[problem_id]
        if experiment_id == -1:
            print(problem)
        experiments.extend(list(itertools.product(*problem)))
    if experiment_id == -1:
        print(len(experiments))
        return
    experiment = experiments[experiment_id]
    args.problem, args.implementation, args.n_qubits, args.n_runs, args.gate_set, args.phys_noise, args.n_shots = experiment

    main(args)

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("problem", nargs="+", type=int, help="Which problem in the batch to run.")
    argparser.add_argument("experiment", type=int, help="Which experiment in the problem to run. (-1 prints amount of experiments)")
    args = argparser.parse_args()
    setup_experiment(args.problem, args.experiment)