from argparse import ArgumentParser, Namespace
from dataclasses import dataclass

from experiments.experimenter import Experimenter, MultipleRunExperimenter

def main(args:Namespace, unknown:list[str]):
    implementation = args.implementation.lower()
    if "linear_growth" in implementation:
        if args.gate_set == "ROT-CNOT":
            from quantumneat.implementations.linear_growth import LinearGrowthConfig
            config = LinearGrowthConfig
        elif args.gate_set == "R-CNOT":
            from quantumneat.implementations.linear_growth import LinearGrowthConfigSeparate
            config = LinearGrowthConfigSeparate
        else:
            raise NotImplementedError(f"Gateset {args.gate_set} not found.")
    elif "qneat" in implementation:
        from quantumneat.implementations.qneat import QNEAT_Config
        config = QNEAT_Config
    else:
        raise NotImplementedError(f"Implementation {implementation} not found.")
    config = config(args.n_qubits, args.population_size, number_of_cpus=args.number_of_cpus)
        
    problem_arg:str = args.problem.lower()
    if "cim" in problem_arg or "classical_ising" in problem_arg:
        from quantumneat.problems.ising import ClassicalIsing
        problem = ClassicalIsing(config)
    elif "tfim" in problem_arg or "transverous_ising" in problem_arg:
        from quantumneat.problems.ising import TransverseIsing
        if "g_" in problem_arg:
            g = float(problem_arg.split("g_")[-1])
        else:
            g = 1
        problem = TransverseIsing(config, g)
    elif "fith" in problem_arg or "fox_in_the_hole" in problem_arg:
        if "gates" in problem_arg:
            from quantumneat.problems.fox_in_the_hole import FoxInTheHoleNGates
            problem = FoxInTheHoleNGates(config)
        else:
            from quantumneat.problems.fox_in_the_hole import FoxInTheHole
            problem = FoxInTheHole(config)
    elif "h2" in problem_arg or "hydrogen" in problem_arg:
        from quantumneat.problems.hydrogen import Hydrogen
        if "r_" in problem_arg:
            distance = float(problem_arg.split("r_")[-1])
            config.h2_distance = distance
        problem = Hydrogen(config)
    else:
        raise NotImplementedError(f"Problem {problem_arg} not found.")
    
    if args.name is None:
        args.name = f"{problem_arg}_{implementation}"
    args.name += f"_{args.gate_set}_{args.n_qubits}-qubits_{args.population_size}-population"
    
    if args.optimizer_steps > 0:
        config.optimize_energy = True
        config.optimize_energy_max_iter = args.optimizer_steps
        args.name += f"_{args.optimizer_steps}-optimizer-steps"

    if "no-force" in args.extra_info:
        config.force_population_size = False
        args.name += "_no-forced-population"
    if "normalise" in args.extra_info:
        config.normalise_fitness = True
        args.name += "_normalised-fitness"
    if "remove-stagnant" in args.extra_info:
        config.remove_stagnant_species = True
        args.name += "_remove-stagnant"
        
    print(args.name)
    if args.n_runs > 0:
        experimenter = MultipleRunExperimenter(args.name, config, problem, folder=".")
        experimenter.run_multiple_experiments(args.n_runs, args.generations, do_plot_individual=False, do_plot_multiple=args.plot, do_print=True)
    else:
        experimenter = Experimenter(args.name, config, problem, folder=".")
        experimenter.run_default(args.generations, do_plot=args.plot, do_print=True)
    
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
    argparser.add_argument("--plot",                    action="store_true",                          help="Whether to plot the results")
    argparser.add_argument("-X", "--extra_info",        type=str, default="",                         help="Extra settings")
    args, unknown = argparser.parse_known_args()
    main(args, unknown)