from dataclasses import dataclass
import numpy as np
import logging

from . import genome as gen
from . import gate as g
from . import helper as h
from . import species as s

@dataclass
class QuantumNEATConfig:
    """
    Class for keeping the configuration settings of the QNEAT algorithm
    """
    # Global settings
    n_qubits: int
    population_size: int
    global_innovation_number = h.GlobalInnovationNumber()
    global_species_number = h.GlobalSpeciesNumber()

    # Main QNEAT settings
    compatibility_threshold: float = 3
    prob_mutation_without_crossover: float = 0.25
    specie_champion_size: int = 5
    percentage_survivors: float = 0.5

    # Species settings
    Species = s.Species
    
    # Genome settings
    Genome = gen.Genome
    prob_weight_mutation: float = 0.8
    prob_weight_perturbation: float = 0.9
    perturbation_amplitude: float = 1
    prob_add_gate_mutation: float = 0.1
    max_add_gate_tries: int = 10

    # Gate settings
    GateGene = g.GateGene
    parameter_amplitude: float = 2*np.pi
    simulator = 'qulacs' # 'qiskit'

    # GateType settings
    GateType = g.GateTypes

class QuantumNEATExperimenterConfig(QuantumNEATConfig):
    # Logger settings
    filename: str = "experiment"
    file_level = logging.INFO
    console_level = logging.ERROR
    mode = "a"