from __future__ import annotations

from time import time
from typing import TYPE_CHECKING
from abc import abstractmethod
import warnings

warnings.filterwarnings("ignore", "Couldn't import `kahypar` - skipping from default hyper optimizer and using basic `labels` method instead.")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import quimb as q

from quantumneat.quant_lib_np import I, Z, ZZ, YY, XX
from quantumneat.problem import Problem
from quantumneat.helper import get_energy_qulacs

if TYPE_CHECKING:
    from quantumneat.configuration import QuantumNEATConfig, Circuit
    from quantumneat.genome import Genome

def exact_diagonalisation(H):
    el, ev = q.eigh(H, k=1)
    return el[0]

H2_DATA = pd.read_csv('H2.csv', index_col=0, sep=',')

def h2_instance(distance = None):
    if distance == None:
        return H2_DATA.sample().iloc[0]
    # print(H2_DATA.loc[distance])
    return  H2_DATA.loc[distance]

def plot_solution(show = False, **plot_kwargs):
    import matplotlib.pyplot as plt
    x = np.array(H2_DATA.index)
    # print(x)
    y = np.zeros(len(x))
    for ind, i in enumerate(x):
        y[ind] = exact_diagonalisation(Hydrogen.hamiltonian(h2_instance(i)))
    # plt.scatter(x, y)
    plt.plot(x,y, **plot_kwargs)
    if show:
        plt.show()

class Hydrogen(Problem):
    def __init__(self, config:QuantumNEATConfig, **kwargs) -> None:
        self.config = config

    def get_instance(self, distance = None) -> tuple[np.ndarray]:
        return h2_instance(distance)
    
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
        total_gradient = 0
        for ind in range(n_parameters):
            temp = parameters[ind]
            parameters[ind] += self.config.epsilon/2
            partial_gradient = self.energy(circuit, parameters, True)
            parameters[ind] -= self.config.epsilon
            partial_gradient -= self.energy(circuit, parameters, True)
            parameters[ind] = temp # Return the parameter to original value
            total_gradient += partial_gradient**2
        return total_gradient/n_parameters

    def energy(self, circuit, parameters, no_optimization = False) -> float:
        # self.logger.debug("H2 energy called", stacklevel=2)
        instance = self.get_instance(self.config.h2_distance)
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
        return expectation
        # self.logger.debug(f"Expectation {expectation}, solution {solution}")
        # return expectation - solution

    @staticmethod
    def hamiltonian(instance:pd.DataFrame) -> list:
        # print("hamiltonian")
        # print(instance)
        H = instance.loc["1"]*I(0, 2) + \
            instance.loc["Z0"]*Z(0, 2) + \
            instance.loc["Z1"]*Z(1, 2) + \
            instance.loc["Z0Z1"]*ZZ(0, 2) + \
            instance.loc["Y0Y1"]*YY(0, 2) + \
            instance.loc["X0X1"]*XX(0, 2)
        # print(H)
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
    
    # def add_encoding_layer(self, circuit):
    #     pass

if __name__ == "__main__":
    plot_solution(True, marker="o")
    