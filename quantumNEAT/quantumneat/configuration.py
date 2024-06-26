from typing import TYPE_CHECKING, TypeVar
from dataclasses import dataclass, field

import numpy as np

from quantumneat.population import Population
from quantumneat.species import Species
from quantumneat.genome import Genome
from quantumneat.gene import Gene
from quantumneat.helper import GlobalInnovationNumber, GlobalSpeciesNumber

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qulacs import ParametricQuantumCircuit
    Circuit = TypeVar('Circuit', QuantumCircuit, ParametricQuantumCircuit)

@dataclass
class QuantumNEATConfig():
    """
    Class for keeping the configuration settings of the QNEAT algorithm
    """
    # Global settings
    n_qubits:int
    population_size:int
    GlobalInnovationNumber = GlobalInnovationNumber()
    GlobalSpeciesNumber = GlobalSpeciesNumber()

    # Main QNEAT settings
    number_of_cpus:int = -1 # Number of cpu's to use for multiprocessing. If < 0 no multiprocessing is used.
    evaluate:bool = True
    calculate_solution:bool = False
    
    # Population settings
    Population = Population
    compatibility_threshold:float = 3
    prob_mutation_without_crossover:float = 0.25
    specie_champion_size:int = 5
    percentage_survivors:float = 0.5
    normalise_fitness:bool = False
    fitness_sharing:bool = False
    force_population_size:bool = True

    # Species settings
    Species = Species
    remove_stagnant_species:bool = False
    stagnant_generation:int = 15
    all_stagnant_n_save:int = 2 # How many species are preserved if all species are stagnant
    
    # Genome settings
    Genome = Genome
    prob_weight_mutation:float = 0.8
    prob_weight_perturbation:float = 0.9
    perturbation_amplitude:float = 1
    prob_add_gene_mutation:float = 0.1
    max_add_gene_tries:int = 10
    simulator:str = 'qulacs' # 'qiskit'
    prevent_gate_duplication:bool = False

    # Problem settings
    optimize_energy:bool = False
    optimize_energy_evaluation:bool = True
    optimize_energy_max_iter:int = 100 # Ignored if optimize_energy == False
    solution_margin:float = 10**-3 # epsilon
    use_total_energy:bool = False # Whether to optimize only one set of parameters for all energies instead of one per energy.

    h2_distance = None
    
    # Gene settings
    gene_types:list[Gene] = field(default_factory=list)
    parameter_amplitude: float = 2*np.pi

    # Helper settings
    epsilon:float = 10**-5
    n_shots:int = -1
    phys_noise:bool = False
    phys_noise_encoding:bool = False
    depolarizing_noise_prob:float = 0.1 # Ignored if phys_noise == False

    # Distance settings
    excess_coefficient:float = 1
    disjoint_coefficient:float = 1
    weight_coefficient:float = 0.4

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self