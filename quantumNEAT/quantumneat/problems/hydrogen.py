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

from quantumneat.quant_lib_np import I, Z, ZZ, YY, XX
from quantumneat.problem import Problem
from quantumneat.helper import get_energy_qulacs, get_energy_qulacs_encoded

if TYPE_CHECKING:
    from quantumneat.configuration import QuantumNEATConfig, Circuit
    from quantumneat.genome import Genome

def exact_diagonalisation(H):
    el, ev = q.eigh(H, k=1)
    return el[0]

H2_DATA = pd.read_csv('H2.csv', index_col=0, sep=',')
# print(H2_DATA.index)
H2_DATA_2 = H2_DATA.copy()
correction = 1.5
H2_DATA_2.iloc[0, 0] -= correction
H2_DATA_2.iloc[1, 0] -= correction
H2_DATA_2.iloc[2, 0] -= correction
H2_DATA_2.iloc[3, 0] -= correction

H2_DATA:pd.DataFrame = pd.read_pickle("hamiltonians/h2_hamiltonian.pkl")
# new_index = []
# for index in H2_DATA.index:
#     new_index.append(np.round(index, 2))
# H2_DATA.insert(0, "R", new_index)
# H2_DATA.reset_index()
# # print(H2_DATA)
# H2_DATA.set_index("R", inplace=True)
# print(H2_DATA)
# print(H2_DATA.index)

def h2_instance(distance = None):
    if distance == None:
        return H2_DATA.sample().iloc[0]
    # print(H2_DATA.loc[distance])
    return  H2_DATA.loc[distance]

def h2_instance_2(distance = None):
    if distance == None:
        return H2_DATA_2.sample().iloc[0]
    # print(H2_DATA.loc[distance])
    return  H2_DATA_2.loc[distance]

def get_solution():
    x = np.array(H2_DATA.index)
    # print(x)
    y = np.zeros(len(x))
    for ind, i in enumerate(x):
        instance = h2_instance(i)
        y[ind] = exact_diagonalisation(Hydrogen.hamiltonian(instance)) + instance["correction"]
    return x, y

def get_solutions(X):
    Y = np.zeros(len(X))
    for ind, i in enumerate(X):
        instance = h2_instance(i)
        Y[ind] = exact_diagonalisation(Hydrogen.hamiltonian(instance)) + instance["correction"]
    return Y

def plot_solution(show = False, **plot_kwargs):
    import matplotlib.pyplot as plt
    x, y = get_solution()
    plt.plot(x,y, **plot_kwargs)
    if show:
        plt.title("Hydrogen molecule")
        plt.ylabel("Energy (a.u.)")
        plt.xlabel("Distance (Angstrom)")
        plt.grid()
        plt.show()

def plot_UCCSD_result(**plot_kwargs):
    import matplotlib.pyplot as plt
    try:
        energies = np.load("UCCSD_H2.npy")
    except:
        print("UCCSD data not found")
        return
    distances = np.array(H2_DATA.index)
    for ind, distance in enumerate(distances):
        energies[ind] += h2_instance(distance)["correction"]
    plt.scatter(distances, energies, **plot_kwargs)

def plot_UCCSD_diff(**plot_kwargs):
    import matplotlib.pyplot as plt
    try:
        energies = np.load("UCCSD_H2.npy")
    except:
        print("UCCSD data not found")
        return
    solution = get_solution()[1]
    distances = np.array(H2_DATA.index)
    for ind, distance in enumerate(distances):
        energies[ind] += h2_instance(distance)["correction"] - solution[ind]
    plt.scatter(distances, energies, **plot_kwargs)

def plot_solution_2(show = False, **plot_kwargs):
    import matplotlib.pyplot as plt
    x = np.array(H2_DATA_2.index)
    # print(x)
    y = np.zeros(len(x))
    for ind, i in enumerate(x):
        y[ind] = exact_diagonalisation(Hydrogen.hamiltonian(h2_instance_2(i)))
    # plt.scatter(x, y)
    plt.plot(x,y, **plot_kwargs)
    if show:
        plt.show()

class Hydrogen(Problem):
    def __init__(self, config:QuantumNEATConfig, error_in_fitness = True, **kwargs) -> None:
        self.config = config
        self.error_in_fitness = error_in_fitness

    def get_instance(self, distance = None) -> tuple[np.ndarray]:
        return h2_instance(distance)
    
    def fitness(self, genome:Genome) -> float:
        circuit, n_parameters = genome.get_circuit()
        parameters = genome.get_parameters()
        gradient = self.gradient(circuit, parameters, n_parameters)
        if self.error_in_fitness:
            circuit_error = genome.get_circuit_error()
        else:
            circuit_error = 0
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
        correction = instance.loc["correction"]
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
        # print("hamiltonian")
        # print(instance)
        # H = instance.loc["1"]*I(0, 2) + \
        #     instance.loc["Z0"]*Z(0, 2) + \
        #     instance.loc["Z1"]*Z(1, 2) + \
        #     instance.loc["Z0Z1"]*ZZ(0, 2) + \
        #     instance.loc["Y0Y1"]*YY(0, 2) + \
        #     instance.loc["X0X1"]*XX(0, 2)
        
        H = instance.loc["II"]*I(0, 2) + \
            instance.loc["ZI"]*Z(0, 2) + \
            instance.loc["IZ"]*Z(1, 2) + \
            instance.loc["ZZ"]*ZZ(0, 2) + \
            instance.loc["XX"]*XX(0, 2)
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
    
    # def add_encoding_layer(self, circuit:Circuit):
    #     pass
    def evaluate(self, circuit:Circuit, parameters, N = 1000):
        max_iter = self.config.optimize_energy_max_iter
        self.config.optimize_energy_max_iter = N
        
        if self.config.h2_distance is None:
            energies = []
            distances = H2_DATA.index
            for distance in distances:
                instance = self.get_instance(distance)
                energies.append(self.energy(circuit, parameters, instance=instance, no_solution=True))
        else:
            distances = [self.config.h2_distance]
            instance = self.get_instance(self.config.h2_distance)
            energies = [self.energy(circuit, parameters, instance=instance, no_solution=True)]
        self.config.optimize_energy_max_iter = max_iter
        return distances, energies
    
class EncodedHydrogen(Hydrogen):
    def get_instance(self, distance=None) -> tuple[ndarray]:
        if distance == None:
            distance = np.random.choice(H2_DATA.index)
        return (super().get_instance(distance), [distance])

    def solution(self, instance = None) -> float:
        if instance is None:
            instance = self.get_instance(self.config.h2_distance)[0]
        hamiltonian = self.hamiltonian(instance)
        return q.eigh(hamiltonian, k=1)[0][0]
    
    def energy(self, circuit, parameters, no_optimization = False, instance = None, no_solution = False) -> float:
        # self.logger.debug("H2 energy called", stacklevel=2)
        if instance is None:
            instance = self.get_instance(self.config.h2_distance)
        instance, enc_params = instance
        hamiltonian = self.hamiltonian(instance)
        correction = instance.loc["correction"]
        if self.config.simulator == 'qulacs':
            def expectation_function(params):
                return get_energy_qulacs_encoded(enc_params,
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
        
        # self.logger.debug(f"Expectation {expectation}, solution {solution}")
        if no_solution:
            return expectation + correction
        solution = exact_diagonalisation(hamiltonian)
        return expectation - solution

    def add_encoding_layer(self, circuit:Circuit):
        super().add_encoding_layer(circuit)
        if self.config.simulator == "qiskit":
            circuit.rx(Parameter('enc_0'), 0)
        elif self.config.simulator == "qulacs":
            circuit.add_parametric_RX_gate(0, -1)

    def evaluate(self, circuit:Circuit, parameters, N = 1000):
        max_iter = self.config.optimize_energy_max_iter
        self.config.optimize_energy_max_iter = N
        energies = []
        for distance in H2_DATA.index:
            instance = self.get_instance(distance)
            energies.append(self.energy(circuit, parameters, instance=instance, no_solution=True))
        self.config.optimize_energy_max_iter = max_iter
        return H2_DATA.index, energies
    
def no_identity_hamiltonian(instance:pd.DataFrame) -> list:
    # print("hamiltonian")
    # print(instance)
    H = instance.loc["Z0"]*Z(0, 2) + \
        instance.loc["Z1"]*Z(1, 2) + \
        instance.loc["Z0Z1"]*ZZ(0, 2) + \
        instance.loc["Y0Y1"]*YY(0, 2) + \
        instance.loc["X0X1"]*XX(0, 2)
    # print(H)
    return H

class Hydrogen_2(Hydrogen):
    def get_instance(self, distance = None) -> tuple[np.ndarray]:
        return h2_instance_2(distance)
    
class EncodedHydrogen_2(EncodedHydrogen):
    def get_instance(self, distance = None) -> tuple[np.ndarray]:
        if distance == None:
            distance = np.random.choice(H2_DATA.index)
        return (h2_instance_2(distance), [distance])

class AllHydrogen(Hydrogen):
    def energy(self, circuit, parameters, no_optimization=False, instance=None, no_solution=False) -> float:
        if instance is not None:
            return super().energy(circuit, parameters, no_optimization, instance, no_solution)
        mean_squared_energy = 0
        distances = H2_DATA.index
        for distance in distances:
            instance = self.get_instance(distance)
            energy = super().energy(circuit, parameters, no_optimization, instance, no_solution)
            mean_squared_energy += energy
        return mean_squared_energy/len(distances)
    
class NoSolutionAllHydrogen(Hydrogen):
    def energy(self, circuit, parameters, no_optimization=False, instance=None, no_solution=False) -> float:
        if instance is not None:
            return super().energy(circuit, parameters, no_optimization, instance, no_solution=True)
        mean_squared_energy = 0
        distances = H2_DATA.index
        for distance in distances:
            instance = self.get_instance(distance)
            energy = super().energy(circuit, parameters, no_optimization, instance, no_solution=True)
            mean_squared_energy += energy
        return mean_squared_energy/len(distances)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plot_UCCSD_result()
    plot_solution(True)
    # plot_solution(True, marker="o")
    # plot_solution_2(True, marker="o")
    plot_UCCSD_diff()
    plt.show()