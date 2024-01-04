from __future__ import annotations

import copy
from typing import TYPE_CHECKING
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from quantumneat.gene import GateGene
from quantumneat.genome import CircuitGenome
from quantumneat.configuration import QuantumNEATConfig
from quantumneat.population import Population
from qulacs import ParametricQuantumCircuit
if TYPE_CHECKING:
    from quantumneat.configuration import Circuit
    from quantumneat.implementations.qneat import QNEAT_Config

class GlobalLayerNumber:
    '''
    Class for keeping a global layer number.
    
    Layer number starts at -1.
    '''
    def __init__(self):
        self._layer_number:int = 0

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
        self.qubits = [qubit%self.config.n_qubits for qubit in self.qubits]
        # self.logger.debug(f"{self.qubits=}")

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
            circuit.rx(Parameter(str(n_parameters)), self.qubits[0])
            n_parameters += 1
            circuit.ry(Parameter(str(n_parameters)), self.qubits[0])
            n_parameters += 1
            circuit.rz(Parameter(str(n_parameters)), self.qubits[0])
            n_parameters += 1
        else:
            raise NotImplementedError(f"Simulation method: {self.config.simulator} not implemented for {self.__class__}")
        return circuit, n_parameters

class LayerGene(GateGene):
    #TODO Look at only adding gates in qubit order

    def __init__(self, config:QNEAT_Config, ind:int) -> None:
        super().__init__(ind, config, range(config.n_qubits))
        self.genes:dict[object, list] = {GateROT:[], GateCNOT:[]}
        self.ind = ind

    def add_gate(self, gate:GateGene) -> bool:
        # self.logger.debug(f"LayerGene.add_gate: {gate=}")
        if type(gate) in self.genes:
            for existing_gate in self.genes[type(gate)]:
                if gate.qubits[0] == existing_gate.qubits[0]:
                    # Don't add the same gate multiple times
                    return False
            self.genes[type(gate)].append(gate)
        else:
            self.genes[type(gate)] = [gate]
        return True
    
    def get_parameters(self):
        parameters = []
        # self.logger.debug(f"{self.genes=}")
        # self.logger.debug(f"{self.genes[GateROT]=}")
        for rotgate in self.genes[GateROT]:
            # self.logger.debug(f"{rotgate.parameters=}")
            parameters.append(rotgate.parameters)
        return parameters

    def add_to_circuit(self, circuit:Circuit, n_parameters):
        for genetype in self.genes.keys():
            for gate in self.genes[genetype]:
                circuit, n_parameters = gate.add_to_circuit(circuit, n_parameters)
        if self.config.simulator == 'qiskit':
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

    def mutate(self):
        # super().mutate() # Don't change the parameters in case of unsuccesful mutation
        if np.random.random() < self.config.prob_add_gene_mutation:
            self.mutate_add_gene()
        if np.random.random() < self.config.prob_weight_mutation:
            self._changed = True
            for gene in self.genes.values():
                gene.mutate_parameters()

    def mutate_add_gene(self):
        innovation_number = self.config.GlobalInnovationNumber.next()
        for _ in range(self.config.max_add_gene_tries):
            new_gene:GateGene = np.random.choice(self.config.gene_types)
            qubits = np.random.choice(range(self.config.n_qubits), 
                                        size = new_gene.n_qubits)
            if len(qubits) == 2:
                qubits[1] = qubits[0] + 1
            new_gene = new_gene(innovation_number, config = self.config, 
                                qubits = qubits)
            if self.add_gene(new_gene):
                break
        else:
            self.config.GlobalInnovationNumber.previous()

    def add_gene(self, gene) -> bool:
        if type(gene) == LayerGene:
            self.genes[gene.ind] = gene
            return True
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
    
    def set_layers(self, layers:LayerGene):
        self.genes = layers

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
        genes = self.genes.values()
        genes = sorted(genes, key=lambda gene: gene.ind)
        for gene in genes:
            circuit, n_parameters = gene.add_to_circuit(circuit, n_parameters)
        self._circuit = circuit
        self._n_circuit_parameters = n_parameters

    def update_gradient(self) -> float:
        super().update_gradient()
        circuit, n_parameters = self.get_circuit()
        parameters = np.array([])
        # self.logger.debug(f"{self.genes.values()=}")
        for gene in self.genes.values():
            # self.logger.debug(f"{gene.get_parameters()=}")
            parameters = np.append(parameters, gene.get_parameters())
        # self.logger.debug(f"{n_parameters==len(parameters)=}; {parameters=}")
        self._gradient = self.config.gradient_function(circuit, n_parameters, 
                                                       parameters, self.config)
        self._energy = self.config.energy_function(circuit, parameters, self.config)
    
    @staticmethod
    def compatibility_distance(genome1:QNEAT_Genome, genome2:QNEAT_Genome, config:QuantumNEATConfig):
        # QNEAT_Genome.logger.debug("compatibility_distance")
        def line_up(genome1:QNEAT_Genome, genome2:QNEAT_Genome):
            matching = 0
            disjoint = 0
            excess = 0
            distances = []

            genes1 = sorted(genome1.genes.values(), key=lambda gene: gene.ind)
            genes2 = sorted(genome2.genes.values(), key=lambda gene: gene.ind)

            # QNEAT_Genome.logger.debug(f"{genes1=}; {genes2=}")

            n_genes1, n_genes2 = len(genes1), len(genes2)
            index1, index2 = 0, 0
            while index1 < n_genes1 or index2 < n_genes2:
                # QNEAT_Genome.logger.debug(f"{index1=}:{n_genes1=}, {index2=}:{n_genes2=}")
                gene1 = genes1[index1] if index1 < n_genes1 else None
                gene2 = genes2[index2] if index2 < n_genes2 else None

                if gene1 and gene2:
                    if gene1.innovation_number < gene2.innovation_number:
                        disjoint += 1
                        index1 += 1
                    elif gene1.innovation_number > gene2.innovation_number:
                        disjoint += 1
                        index2 += 1
                    else:
                        distances.append(gene1.get_distance(gene1, gene2))
                        matching += 1
                        index1 += 1
                        index2 += 1
                elif gene1:
                    excess += n_genes1 - index1
                    break
                elif gene2:
                    excess += n_genes2 - index2
                    break
                else:
                    QNEAT_Genome.logger.warning("Don't think this should ever occur.", exc_info=1)
                    break

            if len(distances) == 0:
                distances = [0]
            return matching, disjoint, excess, np.mean(distances)
        n_genes = max(len(genome1.genes), len(genome2.genes))
        # QNEAT_Genome.logger.debug(f"compatibility_distance {n_genes=}")
        if n_genes == 0:            
            # If both genomes are empty, distance == 0, prevent division by 0.
            return 0
        matching, disjoint, excess, avg_distance = line_up(genome1, genome2)
        return config.excess_coefficient*excess/n_genes + config.disjoint_coefficient*disjoint/n_genes + config.weight_coefficient*avg_distance
    
    @staticmethod
    def crossover(genome1: QNEAT_Genome, genome2: QNEAT_Genome, config:QNEAT_Config) -> QNEAT_Genome:
        # Assumes genome1.genes, genome2.genes are sorted by innovation_number 
        # and equal genes have equal innovation_number.
        # QNEAT_Genome.logger.debug("crossover")
        child = config.Genome(genome1.config)
        if genome1.get_fitness() > genome2.get_fitness():
            better = "genome1"
        elif genome1.get_fitness() < genome2.get_fitness():
            better = "genome2"
        else:
            better = "equal"

        # QNEAT_Genome.logger.debug(f"{genome1.get_fitness()=}; {genome2.get_fitness()=}, {better=}")

        genes1 = sorted(genome1.genes.values(), key=lambda gene: gene.ind)
        genes2 = sorted(genome2.genes.values(), key=lambda gene: gene.ind)
        n_genes1, n_genes2 = len(genes1), len(genes2)
        # QNEAT_Genome.logger.info(f"{n_genes1=}; {n_genes2=}")
        index1, index2 = 0, 0
        while index1 < n_genes1 or index2 < n_genes2:
            # QNEAT_Genome.logger.debug(f"{index1=}:{n_genes1=}, {index2=}:{n_genes2=}")
            gene1 = genes1[index1] if index1 < n_genes1 else None
            gene2 = genes2[index2] if index2 < n_genes2 else None
            chosen_gene = None
            if gene1 and gene2:
                # QNEAT_Genome.logger.debug("gene1 and gene2")
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
                # QNEAT_Genome.logger.debug("gene1 and (not gene2) and better == 'genome1'")
                chosen_gene = gene1
                index1 += 1
            elif gene2 and better == "genome2": # excess
                # QNEAT_Genome.logger.debug("(not gene1) and gene2 and better == 'genome2'")
                chosen_gene = gene2
                index2 += 1
            elif better == "equal":
                if np.random.random() < 0.5:
                    if gene1:
                        chosen_gene = gene1
                        index1 += 1
                    elif gene2:
                        chosen_gene = gene2
                        index2 += 1
            else: # excess
                # QNEAT_Genome.logger.debug("not (gene1 and better == 'genome1') and not (gene2 and better == 'genome2')")
                break
            if chosen_gene: # not None
                if chosen_gene == "random":
                    chosen_gene = np.random.choice([gene1, gene2])
                # QNEAT_Genome.logger.debug(f"crossover: {chosen_gene=}")
                if not child.add_gene(copy.deepcopy(chosen_gene)):
                    QNEAT_Genome.logger.error("Child did not add gene of parent.")

        return child
    
class QNEAT_Population(Population):
    def __init__(self, config: QNEAT_Config) -> None:
        super().__init__(config)
        self.config:QNEAT_Config

    def generate_initial_population(self) -> list[QNEAT_Genome]:
        population = []
        initial_layers = {}
        for _ in range(self.config.initial_layers):
            layer_number = self.config.GlobalLayerNumber.next() - 1
            new_layer = LayerGene(self.config, layer_number)
            for qubit in range(self.config.n_qubits):
                innovation_number = InnovationTracker.get_innovation(layer_number, qubit, 'rot', self.config)
                new_rot = GateROT(innovation_number, self.config, qubits=[qubit])
                new_layer.add_gate(new_rot)
            for qubit in range(self.config.n_qubits):
                innovation_number = InnovationTracker.get_innovation(layer_number, qubit, 'cnot', self.config)
                new_cnot = GateCNOT(innovation_number, self.config, qubits=[qubit, qubit+1])
                new_layer.add_gate(new_cnot)
            initial_layers[layer_number] = new_layer
        for _ in range(self.config.population_size):
            genome = self.config.Genome(self.config)
            # Add self.config.initial_layers amount of full layers initially
            genome.set_layers(initial_layers)
            population.append(genome)
        return self.sort_genomes(population)
    
@dataclass
class QNEAT_Config(QuantumNEATConfig):
    #                                                   # Name in original
    Population = QNEAT_Population
    Genome = QNEAT_Genome
    gene_types:list[GateGene] = field(default_factory=lambda:[GateROT, GateCNOT])
    GlobalLayerNumber = GlobalLayerNumber()       
    initial_layers:int = 0                              # num_initial_layers
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