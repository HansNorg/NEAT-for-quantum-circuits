from argparse import ArgumentParser, Namespace
from dataclasses import dataclass

from experiments.experimenter import Experimenter, MultipleRunExperimenter
from quantumneat.implementations.linear_growth import LinearGrowthConfig, LinearGrowthConfigSeparate
from quantumneat.implementations.qneat import QNEAT_Config

def main(args:Namespace, unknown:list[str]):
    implementation = args.implementation.lower()
    if "linear_growth" in implementation:
        if args.gate_set == "ROT-CNOT":
            config = LinearGrowthConfig
        elif args.gate_set == "R-CNOT":
            config = LinearGrowthConfigSeparate
        else:
            raise NotImplementedError(f"Gateset {args.gate_set} not found.")
    elif "qneat" in implementation:
        config = QNEAT_Config
    else:
        raise NotImplementedError(f"Implementation {implementation} not found.")
    config = config(args.n_qubits, args.population_size, number_of_cpus=args.number_of_cpus)
        
    problem = args.problem.lower()
    if "cim" in problem or "classical_ising" in problem:
        # config.fitness_function
        # config.gradient_function
        # config.energy_function
        print("cim")
    elif "tfim" in problem or "transverous_ising" in problem:
        print("tfim")
    elif "fith" in problem or "fox_in_the_hole" in problem:
        print("fith")
    else:
        raise NotImplementedError(f"Problem {problem} not found.")
    
    if args.name is None:
        args.name = f"{problem}_{implementation}"
    args.name += f"_{args.gate_set}_{args.n_qubits}-qubits_{args.population_size}-population"
    
    if args.optimizer_steps > 0:
        config.optimize_energy = True
        config.optimize_energy_max_iter = args.optimizer_steps
        args.name += f"_{args.optimizer_steps}-optimizer-steps"

    print(args.name)
    if args.n_runs > 0:
        experimenter = MultipleRunExperimenter(args.name, config, folder=".")
        experimenter.run_multiple_experiments(args.n_runs, args.generations, do_plot_individual=True, do_plot_multiple=True, do_print=True)
    else:
        experimenter = Experimenter(args.name, config, folder=".")
        experimenter.run_default(args.generations, do_plot=True, do_print=True)
    
if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("problem",                   type=str,                                     help="which problem to solve")
    argparser.add_argument("implementation",            type=str, choices=["linear_growth", "qneat"], help="which implementation to use")
    argparser.add_argument("--name",                    type=str,                                     help="experiment name")
    argparser.add_argument("-O", "--optimizer_steps",   type=int, default=0,                          help="how many energy optimization steps to do")
    argparser.add_argument("-N", "--n_qubits",          type=int, default=5,                          help="number of qubits")
    argparser.add_argument("-P", "--population_size",   type=int, default=100,                        help="population size")
    argparser.add_argument("-G", "--generations",       type=int, default=100,                        help="amount of generations")
    argparser.add_argument("-R", "--n_runs",            type=int, default=0,                          help="number of runs (<= 0) means 1 run, but no aggregation of results")
    argparser.add_argument("-cpus", "--number_of_cpus", type=int, default=-1,                         help="number of cpus to use")
    argparser.add_argument("-gates", "--gate_set",      type=str, default="ROT-CNOT", choices=["ROT-CNOT", "R-CNOT"], help="which gateset to use")
    args, unknown = argparser.parse_known_args()
    main(args, unknown)