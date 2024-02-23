from __future__ import annotations

from time import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from quantumneat.configuration import QuantumNEATConfig

from quantumneat.problem import Problem
from quantumneat.helper import get_energy_qulacs
from quantumneat.quant_lib_np import from_string

if TYPE_CHECKING:
    from quantumneat.configuration import QuantumNEATConfig, Circuit
    from quantumneat.genome import Genome

class GroundStateEnergy(Problem):
    def __init__(self, config: QuantumNEATConfig, molecule:str, error_in_fitness = True, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.molecule = molecule
        self._load_data()
        self.error_in_fitness = error_in_fitness

    def _load_data(self, fix_index = False):
        try:
            self.data:pd.DataFrame = pd.read_pickle(f"hamiltonians/{self.molecule}_hamiltonian.pkl")
        except FileNotFoundError as exc:
            self.logger.exception(f"Hamiltonian data for {self.molecule} not found.")
            raise exc
        if fix_index:
            new_index = []
            for index in self.data.index:
                new_index.append(np.round(index, 2))
            self.data.insert(0, "R", new_index)
            self.data.reset_index()
            self.data.set_index("R", inplace=True)
        # print(f"Hamiltonian data: {self.data.keys()} \n {self.data.head()}")
    
    def add_encoding_layer(self, circuit:Circuit):
        if self.config.simulator == "qiskit":
            for qubit in range(self.config.n_qubits):
                circuit.h(qubit)
        elif self.config.simulator == "qulacs":
            for qubit in range(self.config.n_qubits):
                circuit.add_H_gate(qubit)

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

    def energy(self, circuit, parameters, no_optimization = False, subtract_solution = False) -> float:
        energy = 0
        for _, instance in self.data.iterrows():
            energy += self.instance_energy(instance, circuit, parameters, no_optimization)
            if subtract_solution:
                energy -= instance["solution"]
        return energy

    def instance_energy(self, instance, circuit, parameters, no_optimization = False):
        hamiltonian = self.hamiltonian(instance)
        correction = instance.loc["correction"]
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
        return expectation + correction

    @staticmethod
    def hamiltonian(instance:pd.DataFrame) -> list:
        H = 0
        for string, const in instance.items():
            if string == "correction" or string == "solution":
                continue
            H += from_string(string)*const
        return H

    def solution(self) -> float:
        return self.data.loc["solution"]
                
    def evaluate(self, circuit:Circuit, parameters, N = 1000):
        max_iter = self.config.optimize_energy_max_iter
        self.config.optimize_energy_max_iter = N
        
        energies = []
        for _, instance in self.data.iterrows():
            energies.append(self.instance_energy(instance, circuit, parameters))

        self.config.optimize_energy_max_iter = max_iter
        return self.data.index, energies
    
    def plot_solution(self, show = False, **plot_kwargs):
        import seaborn as sns
        sns.lineplot(self.data, x="R", y="solution", **plot_kwargs)
        if show:
            import matplotlib.pyplot as plt
            plt.show()

class GroundStateEnergySavedHamiltonian(GroundStateEnergy):
    def __init__(self, config: QuantumNEATConfig, molecule: str, error_in_fitness=True, **kwargs) -> None:
        super().__init__(config, molecule, error_in_fitness, **kwargs)
        self._add_hamiltonian_to_data()

    def _add_hamiltonian_to_data(self):
        hamiltonians = []
        for _, instance in self.data.iterrows():
            hamiltonians.append(super().hamiltonian(instance))
        self.data["hamiltonian"] = hamiltonians
        # print(np.shape(hamiltonians))
    
    @staticmethod
    def hamiltonian(instance):
        return instance["hamiltonian"]

if __name__ == "__main__":
    # problem = GroundStateEnergy(None, "h2")
    problem = GroundStateEnergySavedHamiltonian(None, "lih")
    print(problem.data.head())
    for _, instance in problem.data.iterrows():
        print(problem.hamiltonian(instance))
        break