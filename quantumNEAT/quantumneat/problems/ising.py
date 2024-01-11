from __future__ import annotations

from time import time
from typing import TYPE_CHECKING
from abc import abstractmethod

import numpy as np
from numpy import ndarray
from scipy.optimize import minimize
import quimb as q

from quantumneat.quant_lib_np import X, Z, ZZ
from quantumneat.problem import Problem
from quantumneat.helper import get_energy_qulacs

if TYPE_CHECKING:
    from quantumneat.configuration import QuantumNEATConfig
    from quantumneat.genome import Genome

def ising_1d_instance(n_qubits, seed = None):
    def rand1d(qubits):
        np.random.seed(seed)
        return [np.random.choice([+1, -1]) for _ in range(qubits)]

    # transverse field terms
    h = rand1d(n_qubits)
    # links between lines
    j = rand1d(n_qubits-1)
    return h, j

def exact_diagonalisation(H):
    el, ev = q.eigh(H, k=1)
    return el[0]

class Ising(Problem):
    def get_instance(self, seed = None) -> tuple[np.ndarray]:
        return ising_1d_instance(self.config.n_qubits, seed=seed)
    
    def fitness(self, genome:Genome) -> float:
        circuit, n_parameters = genome.get_circuit()
        parameters = genome.get_parameters()
        gradient = self.gradient(circuit, parameters, n_parameters)
        circuit_error = genome.get_circuit_error()
        energy = self.energy(circuit,parameters)
        # return 1/(1+circuit_error)*gradient
        # return 1/(1+circuit_error)*(-energy)+gradient
        # fitness = 1/(1+circuit_error)-energy+gradient
        # self.logger.debug(f"{gradient=},{circuit_error=},{energy=}, {fitness=}")
        return 1/(1+circuit_error)-energy+gradient

    def energy(self, circuit, parameters, no_optimization = False) -> float:
        instance = self.get_instance()
        hamiltonian = self.hamiltonian(instance)
        solution = exact_diagonalisation(hamiltonian)
        if self.config.simulator == 'qulacs':
            def expectation_function(params):
                return get_energy_qulacs(
                    params, hamiltonian, [], circuit, self.config.n_qubits, 0, 
                    self.config.n_shots, self.config.phys_noise
                )
        else:
            raise NotImplementedError(f"Simulator type {self.config.simulator} not implemented.")
        
        if self.config.optimize_energy and not no_optimization:
            expectation = minimize(
                expectation_function,parameters, method="COBYLA", tol=1e-4, 
                options={'maxiter':self.config.optimize_energy_max_iter}
                ).fun
        else:
            expectation = expectation_function(parameters)
        # return expectation
        return expectation - solution

    @staticmethod
    @abstractmethod
    def hamiltonian(instance) -> list:
        pass

    def solution(self, instance = None) -> float:
        if instance is None:
            instance = self.get_instance()
        hamiltonian = self.hamiltonian(instance)
        return q.eigh(hamiltonian, k=1)[0][0]

    def add_encoding_layer(self, circuit):
        if self.config.simulator == "qiskit":
            for qubit in range(self.config.n_qubits):
                    circuit.h(qubit)
        elif self.config.simulator == "qulacs":
            for qubit in range(self.config.n_qubits):
                    circuit.add_H_gate(qubit)

class ClassicalIsing(Ising):
    @staticmethod
    def hamiltonian(instance):
        h_vec, J_vec = instance[0], instance[1]
        n_qubits = len(h_vec)
        H = 0

        for iq in range(n_qubits -1):
            H += h_vec[iq]*Z(iq, n_qubits) + J_vec[iq]*ZZ(iq, n_qubits)

        H += h_vec[n_qubits-1] * Z(n_qubits-1, n_qubits)

        return H

class TransverseIsing(Ising):
    def __init__(self, config: QuantumNEATConfig, g = 1, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.g = g

    def get_instance(self, seed=None) -> tuple[ndarray]:
        return self.config.n_qubits, self.g

    @staticmethod
    def hamiltonian(instance):
        n_qubits = instance[0]
        H = 0

        for iq in range(n_qubits -1):
            H += instance[1]*X(iq, n_qubits) + ZZ(iq, n_qubits)

        H += instance[1] * X(n_qubits-1, n_qubits)

        return H

class LocalTransverseIsing(Ising):
    @staticmethod
    def hamiltonian(instance):
        h_vec, J_vec = instance[0], instance[1]
        n_qubits = len(h_vec)
        H = 0

        for iq in range(n_qubits -1):
            H += h_vec[iq]*X(iq, n_qubits) + J_vec[iq]*ZZ(iq, n_qubits)

        H += h_vec[n_qubits-1] * X(n_qubits-1, n_qubits)

        return H

def bruteforceLowestValue(h,j):
    def bool_to_state(integer):
    # Convert the 1/0 of a bit to +1/-1
        return 2*int(integer)-1

    r1=list([f'{i:0{len(h)}b}' for i in range(2**len(h))])
    best_energy = np.inf
    best_configurations = []

    for k in range(0,len(r1)):
        current_energy = 0
        # r2[k] is the number of shots that have this result
        # r1[k] is the result as qubits (like 0001)
        # Energy of h
        current_energy += sum([bool_to_state(r1[k][bit_value])*h[bit_value] for bit_value in range(0,len(r1[k]))])
        # Energy of j
        current_energy += sum([bool_to_state(r1[k][bit_value])*bool_to_state(r1[k][bit_value+1])*j[bit_value] for bit_value in range(0,len(j))])
        if current_energy < best_energy:
            best_energy = current_energy
            best_configurations = [r1[k]]
        elif current_energy == best_energy:
            best_configurations.append(r1[k])

    return best_energy, best_configurations

def qubit_to_spin(state):
    def bool_to_state(integer):
        # Convert the 1/0 of a bit to +1/-1
        return 2*int(integer)-1
    return [bool_to_state(state[bit_value]) for bit_value in range(0,len(state))]

if __name__ == "__main__":
    observable_h, observable_j = ising_1d_instance(5, seed = 0)
    print(observable_h, observable_j)
    H = ClassicalIsing.hamiltonian((observable_h, observable_j))
    print(np.shape(H))
    starttime = time()
    el, ev = q.eigh(H, k=1)
    timediff = time() - starttime
    print(el, ev.T, timediff)
    starttime = time()
    exact_classical_energy, exact_classical_configurations = bruteforceLowestValue(observable_h,observable_j)
    timediff = time() - starttime
    print(exact_classical_energy, exact_classical_configurations, [qubit_to_spin(state) for state in exact_classical_configurations], timediff)
    
    # import matplotlib.pyplot as plt
    # import pandas as pd
    # import seaborn as sns
    # exact_classical_energy, exact_classical_configurations = bruteforce_transverse_ising_hamiltonian(observable_h,observable_j)
    # print(exact_classical_energy, exact_classical_configurations, [qubit_to_spin(state) for state in exact_classical_configurations])

    # solutions = pd.DataFrame()
    # for n_qubits in range(2, 11):
    #     solution_lengths = []
    #     length_h_string = int(4*n_qubits)
    #     length_j_string = int(4*(n_qubits-1))
    #     for i in range(1000):
    #         observable_h, observable_j = ising_1d_instance(n_qubits, seed = i)
    #         # observable_h, observable_j = [1, 1, 1, 1, 1], [-1, -1, -1, -1]
    #         # observable_h, observable_j = [1, 1, 1, 1, 1], [1, 1, 1, 1]
    #         # print(observable_h, observable_j)
    #         exact_classical_energy, exact_classical_configurations = bruteforceLowestValue(observable_h,observable_j)
    #         # print(exact_classical_energy, exact_classical_configurations, [qubit_to_spin(state) for state in exact_classical_configurations])
            
    #         # print(f"h = {str(observable_h):{length_h_string}}, j = {str(observable_j):{length_j_string}}, E = {exact_classical_energy:2}, N_solutions = {len(exact_classical_configurations):1}, "+
    #         #     f"qubits = {exact_classical_configurations}, states = {[qubit_to_spin(state) for state in exact_classical_configurations]}")
    #         print(i, end="\r")
    #         solution_lengths.append(len(exact_classical_configurations))
    #     print(f"n_qubits = {n_qubits:2}, avg_solutions = {np.mean(solution_lengths):.3f}, max_solutions = {max(solution_lengths)}")
    #     # plt.hist(solution_lengths, bins=np.arange(0, max(solution_lengths)+1)+0.5, label=f"n_qubits={n_qubits}")
    #     # print(np.histogram(solution_lengths, bins=np.arange(0, max(solution_lengths)+1)+0.5))
    #     # plt.plot(np.arange(1, max(solution_lengths)+1), np.histogram(solution_lengths, bins=np.arange(0, max(solution_lengths)+1)+0.5)[0], label=f"n_qubits={n_qubits}")
    #     solutions[f"{n_qubits}"] = solution_lengths
    # sns.histplot(solutions, multiple="dodge", bins=np.arange(0, max(solution_lengths)+1))
    # plt.xticks(range(1, max(solution_lengths)+1))
    # # plt.legend()
    # plt.show()