from dataclasses import dataclass
import numpy as np
import logging

from quantumneat import genome, gene, helper, species, population

@dataclass
class QuantumNEATConfig:
    """
    Class for keeping the configuration settings of the QNEAT algorithm
    """
    # Global settings
    n_qubits: int
    population_size: int
    global_innovation_number = helper.GlobalInnovationNumber()
    global_species_number = helper.GlobalSpeciesNumber()

    # Main QNEAT settings

    # Population settings
    Population = population.Population
    compatibility_threshold: float = 3
    prob_mutation_without_crossover: float = 0.25
    specie_champion_size: int = 5
    percentage_survivors: float = 0.5

    # Species settings
    Species = species.Species
    
    # Genome settings
    Genome = genome.Genome
    prob_weight_mutation: float = 0.8
    prob_weight_perturbation: float = 0.9
    perturbation_amplitude: float = 1
    prob_add_gene_mutation: float = 0.1
    max_add_gene_tries: int = 10

    # Gene settings
    # Gene = gene.Gene
    GeneTypes = gene.GeneTypes
    parameter_amplitude: float = 2*np.pi
    simulator = 'qulacs' # 'qiskit'

class QuantumNEATExperimenterConfig(QuantumNEATConfig):
    # Logger settings
    filename: str = "experiment"
    file_level = logging.INFO
    console_level = logging.ERROR
    mode = "a"