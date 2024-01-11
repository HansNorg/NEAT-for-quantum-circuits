from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit, QuantumRegister
from qulacs import ParametricQuantumCircuit

from quantumneat.gene import GateGene
from quantumneat.genome import CircuitGenome
from quantumneat.population import Population
from quantumneat.configuration import QuantumNEATConfig
if TYPE_CHECKING:
    from quantumneat.configuration import Circuit
    from quantumneat.implementations.qneat import QNEAT_Config
    from quantumneat.problem import Problem

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
    def __init__(self, innovation_number: int, config: QuantumNEATConfig, problem:Problem, qubits: list[int], **kwargs) -> None:
        super().__init__(innovation_number, config, problem, qubits, **kwargs)
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

    def __init__(self, config:QNEAT_Config, problem:Problem, ind:int) -> None:
        super().__init__(ind, config, problem, qubits=[])
        self.genes:dict[object, list[GateGene]] = {GateROT:[], GateCNOT:[]}
        self.qubits:dict[object, list[int]] = {GateROT:[], GateCNOT:[]}
        self.ind = ind

    def add_gate(self, gate:GateGene) -> bool:
        self.logger.debug(f"add_gate; {type(gate)=}")
        for existing_gate in self.genes[type(gate)]:
            if gate.qubits == existing_gate.qubits:
                # Don't add the same gate multiple times
                return False
        self.genes[type(gate)].append(gate)
        self.qubits[type(gate)].append(gate.qubits[0])
        return True

    def add_to_circuit(self, circuit:Circuit, n_parameters):
        for genetype in self.genes.keys():
            genes = self.genes[genetype]
            genes = sorted(genes, key=lambda gene:gene.qubits[0])
            for gate in genes:
                circuit, n_parameters = gate.add_to_circuit(circuit, n_parameters)
        if self.config.simulator == 'qiskit':
            circuit.barrier()
        return circuit, n_parameters
    
    def get_parameters(self):
        parameters = []
        for rotgate in self.genes[GateROT]:
            parameters.append(rotgate.parameters)
        return parameters
    
    def gates(self):
        for key in self.genes.keys():
            for gate in self.genes[key]: 
                yield gate

class QNEAT_Genome(CircuitGenome):
    def __init__(self, config:QNEAT_Config) -> None:
        super().__init__(config)
        self.config:QNEAT_Config
        self.genes:dict[int,LayerGene] = {}

    def mutate_add_gene(self):
        gene_type:GateGene = np.random.choice(self.config.gene_types)
        valid = []
        for layer in self.genes.values():
            for qubit in range(self.config.n_qubits):
                if qubit not in layer.genes[gene_type]:
                    valid.append((layer, qubit))
        gene_added = False
        if valid:
            layer, qubit = np.random.choice(valid)
            if gene_type == GateROT:
                if layer.genes[GateCNOT]:
                    innovation_number = InnovationTracker.get_innovation(layer.ind, qubit, gene_type, self.config)
                    new_gene = GateROT(innovation_number, self.config, self.problem, [qubit])
                    gene_added = layer.add_gate(new_gene)
            elif gene_type == GateCNOT:
                if layer.ind-1 in self.genes.keys():
                    if self.genes[layer.ind-1].genes[GateROT]:
                        innovation_number = InnovationTracker.get_innovation(layer.ind, qubit, gene_type, self.config)
                        new_gene = GateCNOT(innovation_number, self.config, self.problem, [qubit, qubit+1])
                        gene_added = layer.add_gate(new_gene)
        return gene_added
        
    def add_gene(self, gene:LayerGene) -> bool:
        if gene.ind in self.genes:
            return False
        self.genes[gene.ind] = gene
        self._changed()
        return True
        # ind = np.random.randint(self.config.GlobalLayerNumber.current() + 1)
        # if ind == self.config.GlobalLayerNumber.current():
        #     self.config.GlobalLayerNumber.next()
        # if ind not in self.genes.keys():
        #     new_layer = LayerGene(self.config, ind)
        #     self.genes[ind] = new_layer
        # gene_added = self.genes[ind].add_gate(gene)
        # if gene_added:
        #     self._changed()
        # return gene_added

    def update_circuit(self):
        super().update_circuit()
        n_parameters = 0
        if self.config.simulator == "qiskit":
            circuit = QuantumCircuit(QuantumRegister(self.config.n_qubits))
            for qubit in range(self.config.n_qubits):
                circuit.h(qubit)
        elif self.config.simulator == "qulacs":
            circuit = ParametricQuantumCircuit(self.config.n_qubits)
            for qubit in range(self.config.n_qubits):
                circuit.add_H_gate(qubit)
        for gene in self.genes.values(): # Should be sorted first imo, but original doesn't
            circuit, n_parameters = gene.add_to_circuit(circuit, n_parameters)
        self._circuit = circuit
        self._n_circuit_parameters = n_parameters
    
    def update_gradient(self) -> float:
        super().update_gradient()
        circuit, n_parameters = self.get_circuit()
        parameters = np.array([])
        for gene in self.genes.values():
            parameters = np.append(parameters, gene.get_parameters())
        self._gradient = self.config.gradient_function(circuit, n_parameters, 
                                                       parameters, self.config)
        
    @staticmethod
    def crossover(genome1: QNEAT_Genome, genome2: QNEAT_Genome) -> QNEAT_Genome:
        # Assumes genome1.genes, genome2.genes are sorted by innovation_number 
        # and equal genes have equal innovation_number.
        QNEAT_Genome.logger.debug("crossover")
        child = CircuitGenome(genome1.config)
        
        for gene_type in genome1.config.gene_types:
            n_genes1, n_genes2 = len(genome1.genes[gene_type]), len(genome2.genes[gene_type])
            if genome1.get_fitness() > genome2.get_fitness():
                better = "genome1"
            elif genome1.get_fitness() < genome2.get_fitness():
                better = "genome2"
            else:
                if n_genes1 < n_genes2:
                    better = "genome1"
                elif n_genes1 > n_genes2:
                    better = "genome2"
                else:
                    better = np.random.choice(["genome1", "genome2"])
            QNEAT_Genome.logger.debug(f"{genome1.get_fitness()=}, {genome2.get_fitness()=}, {better=}")

            index1, index2 = 0, 0
            while index1 < n_genes1 or index2 < n_genes2:
                QNEAT_Genome.logger.debug(f"{index1=}:{n_genes1=}, {index2=}:{n_genes2=}")
                gene1 = genome1.genes[gene_type][index1] if index1 < n_genes1 else None
                gene2 = genome2.genes[gene_type][index2] if index2 < n_genes2 else None
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
                for qubit in range(self.config.n_qubits):
                    innovation_number = InnovationTracker.get_innovation(layer_number, qubit, 'rot', self.config)
                    new_rot = GateROT(innovation_number, self.config, qubits=[qubit])
                    new_layer.add_gate(new_rot)
                for qubit in range(self.config.n_qubits):
                    innovation_number = InnovationTracker.get_innovation(layer_number, qubit, 'cnot', self.config)
                    new_cnot = GateCNOT(innovation_number, self.config, qubits=[qubit, qubit+1])
                    new_layer.add_gate(new_cnot)
            population.append(genome)
        return self.sort_genomes(population)
    
@dataclass
class QNEAT_Config(QuantumNEATConfig):
    #                                                   # Name in original
    Population = QNEAT_Population
    Genome = QNEAT_Genome
    gene_types:list[GateGene] = field(default_factory=lambda:[GateROT, GateCNOT])
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