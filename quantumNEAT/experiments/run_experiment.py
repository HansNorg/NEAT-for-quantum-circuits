from argparse import ArgumentParser, Namespace
from dataclasses import dataclass

from experiments.experimenter import Experimenter, MultipleRunExperimenter

cluster_n_shots = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

def main(args:Namespace, unknown:list[str] = []):
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
    config = config(args.n_qubits, args.population_size, 
                    number_of_cpus=args.number_of_cpus, 
                    simulator=args.simulator,
                    n_shots = args.n_shots,
                    phys_noise=args.phys_noise,
                    )
    if args.total_energy:
        config.use_total_energy = True
    if args.fitness_sharing:
        config.fitness_sharing = True
    
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
    elif "gs" in problem_arg:
        config.evaluate = True
        error_in_fitness = True
        if "errorless" in problem_arg:
            error_in_fitness = False
        if "h2" in problem_arg:
            molecule = "h2"
        elif "h6" in problem_arg:
            molecule = "h6"
        elif "lih" in problem_arg:
            molecule = "lih"
        if "saveh" in problem_arg:
            from quantumneat.problems.chemistry import GroundStateEnergySavedHamiltonian as GroundStateEnergy
        else:
            from quantumneat.problems.chemistry import GroundStateEnergy
        problem = GroundStateEnergy(config, molecule, error_in_fitness=error_in_fitness)
        if "hf" in problem_arg:
            # from quantumneat.problems.chemistry import GroundStateEnergy.add_hartree_fock_encoding
            problem.add_encoding_layer = problem.add_hartree_fock_encoding
        elif "0" in problem_arg:
            problem.add_encoding_layer = problem.no_encoding_layer
        if "no-fitness" in problem_arg:
            problem.fitness = problem.no_fitness
        elif "random" in problem_arg:
            problem.fitness = problem.random_fitness
    elif "h2" in problem_arg or "hydrogen" in problem_arg:
        error_in_fitness = True
        if "errorless" in problem_arg:
            error_in_fitness = False
        if "r_" in problem_arg:
            distance = float(problem_arg.split("r_")[-1])
            config.h2_distance = distance
        config.evaluate = True
        if "all" in problem_arg:
            if "no-solution" in problem_arg:
                from quantumneat.problems.hydrogen import NoSolutionAllHydrogen
                problem = NoSolutionAllHydrogen(config, error_in_fitness=error_in_fitness)
            else:
                from quantumneat.problems.hydrogen import AllHydrogen
                problem = AllHydrogen(config, error_in_fitness=error_in_fitness)
        else:
            if "encoded" in problem_arg:
                from quantumneat.problems.hydrogen import EncodedHydrogen
                problem = EncodedHydrogen(config, error_in_fitness=error_in_fitness)
            else:
                from quantumneat.problems.hydrogen import Hydrogen
                problem = Hydrogen(config, error_in_fitness=error_in_fitness)
        if "no-identity" in problem_arg:
            from quantumneat.problems.hydrogen import no_identity_hamiltonian
            problem.hamiltonian = no_identity_hamiltonian
    elif "h6" in problem_arg:
        if "errorless" in problem_arg:
            error_in_fitness = False
        if "r_" in problem_arg:
            distance = float(problem_arg.split("r_")[-1])
            config.h2_distance = distance
        config.evaluate = True
        if "all" in problem_arg:
            if "no-solution" in problem_arg:
                from quantumneat.problems.hydrogen_6 import NoSolutionAllHydrogen6
                problem = NoSolutionAllHydrogen6(config, error_in_fitness=error_in_fitness)
            else:
                from quantumneat.problems.hydrogen_6 import AllHydrogen6
                problem = AllHydrogen6(config)
        else:
            from quantumneat.problems.hydrogen_6 import Hydrogen6
            problem = Hydrogen6(config)
    elif "lih" in problem_arg or "lithium_hydride" in problem_arg:
        if "errorless" in problem_arg:
            error_in_fitness = False
        if "r_" in problem_arg:
            distance = float(problem_arg.split("r_")[-1])
            config.h2_distance = distance
        config.evaluate = True
        if "all" in problem_arg:
            if "no-solution" in problem_arg:
                from quantumneat.problems.lithium_hydride import NoSolutionAllLithiumHydride
                problem = NoSolutionAllLithiumHydride(config, error_in_fitness=error_in_fitness)
            else:
                from quantumneat.problems.lithium_hydride import AllLithiumHydride
                problem = AllLithiumHydride(config)
        if "part" in problem_arg:
            if "no-solution" in problem_arg:
                from quantumneat.problems.lithium_hydride import NoSolutionPartLithiumHydride
                problem = NoSolutionPartLithiumHydride(config, error_in_fitness=error_in_fitness)
            else:
                # from quantumneat.problems.lithium_hydride import AllLithiumHydride
                # problem = AllLithiumHydride(config)
                raise NotImplementedError(f"Problem {problem_arg} not implemented.")
        else:
            from quantumneat.problems.lithium_hydride import LithiumHydride
            problem = LithiumHydride(config)
    else:
        raise NotImplementedError(f"Problem {problem_arg} not found.")
    
    if args.name is None:
        args.name = ""
    args.name += f"{problem_arg}_{implementation}"
    args.name += f"_{args.gate_set}_{args.n_qubits}-qubits_{args.population_size}-population"
    
    if args.optimizer_steps > 0:
        config.optimize_energy = True
        config.optimize_energy_max_iter = args.optimizer_steps
        args.name += f"_{args.optimizer_steps}-optimizer-steps"

    if args.total_energy:
        args.name += "_total-energy"
    if args.fitness_sharing:
        args.name += "_shared-fitness"

    args.name += f"_{args.n_shots}-shots"
    if args.phys_noise:
        args.name += "_phys-noise"

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
    argparser.add_argument("-cpus", "--number_of_cpus", type=int, default=-1,                         help="number of cpus to use for explicit multiprocessing. (-1 means no multiprocessing)")
    argparser.add_argument("-gates", "--gate_set",      type=str, default="ROT-CNOT", choices=["ROT-CNOT", "R-CNOT"], help="which gateset to use")
    argparser.add_argument("--plot",                    action="store_true",                          help="Whether to plot the results")
    argparser.add_argument("--phys_noise",              action="store_true",                          help="Whether to add physical noise in the simulation")
    argparser.add_argument("--n_shots",                 type=int, default=0,                         help="How many shots are taken for shot noise. (0 means no shot noise)")
    argparser.add_argument("--simulator",               type=str, default="qulacs",                   help="Which software package to use for simulation of circuits")
    argparser.add_argument("--total_energy",            action="store_true",                          help="Whether to optimize only one set of parameters for all energies instead of one per energy.")
    argparser.add_argument("--fitness_sharing",         action="store_true",                          help="Whether to use fitness sharing.")
    argparser.add_argument("-X", "--extra_info",        type=str, default="",                         help="Extra settings")
    args, unknown = argparser.parse_known_args()
    args.n_shots = cluster_n_shots[args.n_shots]
    main(args, unknown)