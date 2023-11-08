from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Union

import numpy as np
from qiskit.circuit import Parameter

from quantumneat.gene import GateGene
from quantumneat.genome import CircuitGenome
from quantumneat.population import Population
from quantumneat.configuration import QuantumNEATConfig
if TYPE_CHECKING:
    from quantumneat.configuration import Circuit
    from quantumneat.implementations.qneat import QNEAT_Config

class GlobalLayerNumber:
    '''
    Class for keeping a global layer number.
    
    Layer number starts at 0.
    '''
    def __init__(self):
        self._layer_number:int = -1

    def next(self):
        '''
        Get the next layer number.

        Increments the layer number.
        '''
        self._layer_number += 1
        return self._layer_number
    
    def current(self):
        return self._layer_number
    
class InnovationTracker:            # Original: Gate/InnovTable
    gate_history   = {'rot': {}, 'cnot': {}}

    @staticmethod
    def get_innovation(layer_number:int, qubit:int, type:str, config:QNEAT_Config):
        if not InnovationTracker.gate_history[type].get((layer_number, qubit),False):
            InnovationTracker.gate_history[type][layer_number, qubit] = config.GlobalInnovationNumber.next()
        return InnovationTracker.gate_history[type][layer_number, qubit]        

class GateCNOT(GateGene):
    n_qubits = 2
    def __init__(self, innovation_number: int, config: QuantumNEATConfig, qubits: list[int], **kwargs) -> None:
        super().__init__(innovation_number, config, qubits, **kwargs)
        self.qubits = self.qubits%self.config.n_qubits

    def add_to_circuit(self, circuit:Circuit, n_parameters:int) -> tuple[Circuit, int]:
        if self.config.simulator == 'qulacs':
            circuit.add_CNOT_gate(self.qubits[0], self.qubits[1])
        elif self.config.simulator == 'qiskit':
            circuit.cnot(self.qubits[0], self.qubits[1])
        else:
            raise NotImplementedError(f"Simulation method: {self.config.simulator} not implemented for {self.__class__}")
        return circuit, n_parameters
    
class GateROT(GateGene):
    n_qubits = 1
    n_parameters = 3
    
    def add_to_circuit(self, circuit:Circuit, n_parameters:int) -> tuple[Circuit, int]:
        if self.config.simulator == 'qulacs':
            circuit.add_parametric_RX_gate(self.qubits[0], self.parameters[0])
            circuit.add_parametric_RY_gate(self.qubits[0], self.parameters[1])
            circuit.add_parametric_RZ_gate(self.qubits[0], self.parameters[2])
            n_parameters += 3
        elif self.config.simulator == 'qiskit':
            circuit.rx(Parameter(n_parameters), self.qubits[0])
            n_parameters += 1
            circuit.ry(Parameter(n_parameters), self.qubits[0])
            n_parameters += 1
            circuit.rz(Parameter(n_parameters), self.qubits[0])
            n_parameters += 1
        else:
            raise NotImplementedError(f"Simulation method: {self.config.simulator} not implemented for {self.__class__}")
        return circuit, n_parameters

class LayerGene(GateGene):
    #TODO Look at only adding gates in qubit order

    def __init__(self, config:QNEAT_Config, ind:int) -> None:
        super().__init__(ind, config, range(config.n_qubits))
        self.genes:dict[object, list[GateGene]] = {GateROT:[], GateCNOT:[]}
        self.ind = ind

    def add_gate(self, gate:GateGene) -> bool:
        self.logger.debug(f"add_gate; {type(gate)=}")
        for existing_gate in self.genes[type(gate)]:
            if gate.qubits == existing_gate.qubits:
                # Don't add the same gate multiple times
                return False
        self.genes[type(gate)].append(gate)
        return True

    def add_to_circuit(self, circuit:Circuit, n_parameters):
        for genetype in self.config.gene_types:
            if genetype.name in self.genes:
                for gate in self.genes[genetype.name]:
                    circuit, n_parameters = gate.add_to_circuit(circuit, n_parameters)
        circuit.barrier(self.qubits)
        return circuit, n_parameters
    
    def gates(self):
        for key in self.genes.keys():
            for gate in self.genes[key]: 
                yield gate

class QNEAT_Genome(CircuitGenome):
    def __init__(self, config:QNEAT_Config) -> None:
        super().__init__(config)
        self.config:QNEAT_Config
        self.genes:dict[int,LayerGene] = {}

    def add_gene(self, gene) -> bool:
        ind = np.random.randint(self.config.GlobalLayerNumber.current() + 1)
        if ind == self.config.GlobalLayerNumber.current():
            self.config.GlobalLayerNumber.next()
        if ind not in self.genes.keys():
            new_layer = LayerGene(self.config, ind)
            self.genes[ind] = new_layer
        gene_added = self.genes[ind].add_gate(gene)
        if gene_added:
            self._update_fitness = True
        return gene_added
    
    @staticmethod
    def crossover(genome1: QNEAT_Genome, genome2: QNEAT_Genome) -> QNEAT_Genome:
        # Assumes genome1.genes, genome2.genes are sorted by innovation_number 
        # and equal genes have equal innovation_number.
        QNEAT_Genome.logger.debug("crossover")
        child = CircuitGenome(genome1.config)
        if genome1.get_fitness() > genome2.get_fitness():
            better = "genome1"
        elif genome1.get_fitness() < genome2.get_fitness():
            better = "genome2"
        else:
            better = "equal"
        QNEAT_Genome.logger.debug(f"{genome1.get_fitness()=}, {genome2.get_fitness()=}, {better=}")

        n_genes1, n_genes2 = len(genome1.genes), len(genome2.genes)
        index1, index2 = 0, 0
        while index1 < n_genes1 or index2 < n_genes2:
            QNEAT_Genome.logger.debug(f"{index1=}:{n_genes1=}, {index2=}:{n_genes2=}")
            gene1 = genome1.genes[index1] if index1 < n_genes1 else None
            gene2 = genome2.genes[index2] if index2 < n_genes2 else None
            chosen_gene = None
            if gene1 and gene2:
                QNEAT_Genome.logger.debug("gene1 and gene2")
                if gene1.innovation_number < gene2.innovation_number: # disjoint
                    if better == "genome1":
                        chosen_gene = gene1
                    index1 += 1
                elif gene1.innovation_number > gene2.innovation_number: # disjoint
                    if better == "genome2":
                        chosen_gene = gene2
                    index2 += 1
                else: # matching (gene1.innovation_number == gene2.innovation_number)
                    # if better == "genome1":
                    #     chosen_gene = gene1
                    # elif better == "genome2":
                    #     chosen_gene = gene2
                    # else: # better == "equal"
                    #     chosen_gene = "random"
                    chosen_gene = "random"
                    index1 += 1
                    index2 += 1
            elif gene1 and better == "genome1": # excess
                QNEAT_Genome.logger.debug("gene1 and (not gene2) and better == 'genome1'")
                chosen_gene = gene1
                index1 += 1
            elif gene2 and better == "genome2": # excess
                QNEAT_Genome.logger.debug("(not gene1) and gene2 and better == 'genome2'")
                chosen_gene = gene2
                index2 += 1
            elif better == "equal":
                if np.random.random() < 0.5:
                    if gene1: chosen_gene = gene1
                    elif gene2: chosen_gene = gene2
            else: # excess
                QNEAT_Genome.logger.debug("not (gene1 and better == 'genome1') and not (gene2 and better == 'genome2')")
                break
            if chosen_gene: # not None
                if chosen_gene == "random":
                    chosen_gene = np.random.choice([gene1, gene2])
                if not child.add_gene(copy.deepcopy(chosen_gene)):
                    QNEAT_Genome.logger.error("Child did not add gene of parent.")

        return child

class QNEAT_Population(Population):
    def __init__(self, config: QNEAT_Config) -> None:
        super().__init__(config)
        self.config:QNEAT_Config

    def generate_initial_population(self) -> list[QNEAT_Genome]:
        population = []
        for _ in range(self.config.population_size):
            genome = self.config.Genome(self.config)
            # Add self.config.initial_layers amount of full layers initially
            for _ in range(self.config.initial_layers):
                layer_number = self.config.GlobalLayerNumber.next()
                new_layer = LayerGene(self.config, layer_number)
                for qubit in self.config.n_qubits:
                    innovation_number = InnovationTracker.get_innovation(layer_number, qubit, 'rot', self.config)
                    new_rot = GateROT(innovation_number, self.config, qubits=[qubit])
                    new_layer.add_gate(new_rot)
                for qubit in self.config.n_qubits:
                    innovation_number = InnovationTracker.get_innovation(layer_number, qubit, 'cnot', self.config)
                    new_cnot = GateCNOT(innovation_number, self.config, qubits=[qubit, qubit+1])
                    new_layer.add_gate(new_cnot)
            population.append(genome)
        return self.sort_genomes(population)
    
class QNEAT_Config(QuantumNEATConfig):
    #                                                   # Name in original
    Population = QNEAT_Population
    Genome = QNEAT_Genome
    gene_types:list[GateGene] = [GateROT, GateCNOT]  
    GlobalLayerNumber = GlobalLayerNumber()       
    initial_layers:int = 1                              # num_initial_layers
    disjoint_coefficient:float = 1                      # disjoint_coeff
    excess_coefficient:float = disjoint_coefficient     # (Excess and disjoint are not distinguished in original)
    weight_coefficient:float = 1                        # weight_coeff
    prob_add_gene_mutation:float = 0.02                 # add_rot_prob, add_cnot_prob
    prob_weight_mutation:float = 0.1                    # weight_mutate_prob
    prob_weight_perturbation:float = 0.1                # new_weight_prob
    parameter_amplitude:float = np.pi                   # weight_init_range
    perturbation_amplitude:float = 0.5                  # weight_mutate_power
    compatibility_threshold:float = 1                   # compatibility_threshold
    dynamic_compatibility_threshold:bool = True         # dynamic_compatibility_threshold
    percentage_survivors:float = 0.2                    # survival_rate
    max_add_gene_tries:int = 3                          # tries_tournament_selection