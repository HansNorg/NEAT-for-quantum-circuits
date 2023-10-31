from typing import TYPE_CHECKING, TypeVar
from dataclasses import dataclass, field
import logging

import numpy as np

from quantumneat.population import Population
from quantumneat.species import Species
from quantumneat.genome import Genome
from quantumneat.gene import Gene
from quantumneat.helper import Singleton, GlobalInnovationNumber, GlobalSpeciesNumber, get_gradient

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qulacs import ParametricQuantumCircuit
    Circuit = TypeVar('Circuit', QuantumCircuit, ParametricQuantumCircuit)

@dataclass
class QuantumNEATConfig(metaclass = Singleton):
    """
    Class for keeping the configuration settings of the QNEAT algorithm
    """
    # Global settings
    n_qubits: int
    population_size: int
    GlobalInnovationNumber = GlobalInnovationNumber()
    GlobalSpeciesNumber = GlobalSpeciesNumber()

    # Main QNEAT settings

    # Population settings
    Population = Population
    compatibility_threshold: float = 3
    prob_mutation_without_crossover: float = 0.25
    specie_champion_size: int = 5
    percentage_survivors: float = 0.5

    # Species settings
    Species = Species
    
    # Genome settings
    Genome = Genome
    prob_weight_mutation: float = 0.8
    prob_weight_perturbation: float = 0.9
    perturbation_amplitude: float = 1
    prob_add_gene_mutation: float = 0.1
    max_add_gene_tries: int = 10
    simulator = 'qulacs' # 'qiskit'
    gradient_function = get_gradient

    # Gene settings
    gene_types:list[Gene] = field(default_factory=list)
    parameter_amplitude: float = 2*np.pi

    # Helper settings
    epsilon = 10**-5
    n_shots = 1024
    phys_noise = False

    # Distance settings
    c1:float = 1
    c2:float = 1
    c3:float = 0.4