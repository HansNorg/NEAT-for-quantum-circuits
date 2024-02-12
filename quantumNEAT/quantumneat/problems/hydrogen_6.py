from __future__ import annotations

from time import time
from typing import TYPE_CHECKING
from abc import abstractmethod
import warnings

from numpy import ndarray

warnings.filterwarnings("ignore", "Couldn't import `kahypar` - skipping from default hyper optimizer and using basic `labels` method instead.")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import quimb as q
from qiskit.circuit import Parameter

from quantumneat.quant_lib_np import from_string
from quantumneat.problem import Problem
from quantumneat.helper import get_energy_qulacs, get_energy_qulacs_encoded

if TYPE_CHECKING:
    from quantumneat.configuration import QuantumNEATConfig, Circuit
    from quantumneat.genome import Genome

def exact_diagonalisation(H):
    el, ev = q.eigh(H, k=1)
    return el[0]

DATA:pd.DataFrame = pd.read_pickle("h6_hamiltonian.pkl")
new_index = []
for index in DATA.index:
    new_index.append(np.round(index, 2))
DATA.insert(0, "R", new_index)
DATA.reset_index()
# print(H2_DATA)
DATA.set_index("R", inplace=True)
# print(H2_DATA)
# print(H2_DATA.index)

def h6_instance(distance = None):
    if distance == None:
        return DATA.sample().iloc[0]
    # print(H2_DATA.loc[distance])
    return  DATA.loc[distance]

def get_solution():
    x = np.array(DATA.index)
    # print(x)
    y = np.zeros(len(x))
    for ind, i in enumerate(x):
        instance = h6_instance(i)
        y[ind] = exact_diagonalisation(Hydrogen6.hamiltonian(instance)) + instance["repulsion"]
    return x, y

def get_solutions(X):
    Y = np.zeros(len(X))
    for ind, i in enumerate(X):
        instance = h6_instance(i)
        Y[ind] = exact_diagonalisation(Hydrogen6.hamiltonian(instance)) + instance["repulsion"]
    return Y

def plot_solution(show = False, **plot_kwargs):
    import matplotlib.pyplot as plt
    x, y = get_solution()
    plt.plot(x,y, **plot_kwargs)
    if show:
        plt.show()

class Hydrogen6(Problem):
    def __init__(self, config:QuantumNEATConfig, **kwargs) -> None:
        self.config = config

    def get_instance(self, distance = None) -> tuple[np.ndarray]:
        return h6_instance(distance)
    
    def fitness(self, genome:Genome) -> float:
        circuit, n_parameters = genome.get_circuit()
        parameters = genome.get_parameters()
        gradient = self.gradient(circuit, parameters, n_parameters)
        circuit_error = genome.get_circuit_error()
        energy = genome.get_energy()
        return 1/(1+circuit_error)-energy+gradient
    
    def gradient(self, circuit, parameters, n_parameters) -> float:
        if n_parameters == 0:
            return 0 # Prevent division by 0
        instance = self.get_instance(self.config.h2_distance)
        total_gradient = 0
        for ind in range(n_parameters):
            temp = parameters[ind]
            parameters[ind] += self.config.epsilon/2
            partial_gradient = self.energy(circuit, parameters, True, instance)
            parameters[ind] -= self.config.epsilon
            partial_gradient -= self.energy(circuit, parameters, True, instance)
            parameters[ind] = temp # Return the parameter to original value
            total_gradient += partial_gradient**2
        return total_gradient/n_parameters

    def energy(self, circuit, parameters, no_optimization = False, instance = None, no_solution = False) -> float:
        # self.logger.debug("H2 energy called", stacklevel=2)
        if instance is None:
            instance = self.get_instance(self.config.h2_distance)
        hamiltonian = self.hamiltonian(instance)
        correction = instance.loc["repulsion"]
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
        if no_solution:
            return expectation + correction
        return expectation - solution

    @staticmethod
    def hamiltonian(instance:pd.DataFrame) -> list:
        H = 0
        for string, const in instance.items():
            if string == "repulsion":
                continue
            H += from_string(string)*const
        return H

    def solution(self, instance = None) -> float:
        if instance is None:
            instance = self.get_instance(self.config.h2_distance)
        hamiltonian = self.hamiltonian(instance)
        return q.eigh(hamiltonian, k=1)[0][0]

    def add_encoding_layer(self, circuit:Circuit):
        if self.config.simulator == "qiskit":
            for qubit in range(self.config.n_qubits):
                circuit.h(qubit)
        elif self.config.simulator == "qulacs":
            for qubit in range(self.config.n_qubits):
                circuit.add_H_gate(qubit)
    
    # def add_encoding_layer(self, circuit:Circuit):
    #     pass
    def evaluate(self, circuit:Circuit, parameters, N = 1000):
        max_iter = self.config.optimize_energy_max_iter
        self.config.optimize_energy_max_iter = N
        
        if self.config.h2_distance is None:
            energies = []
            distances = DATA.index
            for distance in distances:
                instance = self.get_instance(distance)
                energies.append(self.energy(circuit, parameters, instance=instance, no_solution=True))
        else:
            distances = [self.config.h2_distance]
            instance = self.get_instance(self.config.h2_distance)
            energies = [self.energy(circuit, parameters, instance=instance, no_solution=True)]
        self.config.optimize_energy_max_iter = max_iter
        return distances, energies
    
class AllHydrogen6(Hydrogen6):
    def energy(self, circuit, parameters, no_optimization=False, instance=None, no_solution=False) -> float:
        if instance is not None:
            return super().energy(circuit, parameters, no_optimization, instance, no_solution)
        mean_squared_energy = 0
        distances = DATA.index
        for distance in distances:
            instance = self.get_instance(distance)
            energy = super().energy(circuit, parameters, no_optimization)
            mean_squared_energy += energy**2
        return mean_squared_energy/len(distances)
    
if __name__ == "__main__":
    print(DATA.columns)
    plot_solution(True, marker="o")
    # plot_solution_2(True, marker="o")
    