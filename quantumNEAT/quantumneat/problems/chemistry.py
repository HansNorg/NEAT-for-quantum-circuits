from __future__ import annotations

from time import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from quantumneat.configuration import QuantumNEATConfig

from quantumneat.problem import Problem
from quantumneat.helper import get_energy_qulacs, get_energy_qiskit, get_energy_qiskit_no_transpilation
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
    
    def print_hamiltonian(self):
        # print(f"Hamiltonian data: {self.data.keys()} \n {self.data.head()}")
        print("&\Ham(R) = ", end="\\\\&")
        for ind, key in enumerate(self.data.keys()):
            if key == "solution" or key == "correction" or key == "hamiltonian":
                continue
            if ind != 0:
                print(f" + ", end="")
                if ind % 5 == 0:
                    print("\\\\&", end="")
            print("c_{" + str(ind) + "}(R) ", end="")
            for qubit, op in enumerate(key):
                if op != "I":
                    print("\\"+str(op)+"_{"+ str(qubit)+"}", end="")
        print()
    
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
        # gradient = self.gradient(circuit, parameters, n_parameters)
        gradient = genome.get_gradient()
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
        noise_weights = np.ones(self.config.n_qubits)
        if self.config.simulator == 'qulacs':
            def expectation_function(params):
                return get_energy_qulacs(
                    params, hamiltonian, noise_weights, circuit, self.config.n_qubits, 0, 
                    self.config.n_shots, self.config.phys_noise
                )
        # elif self.config.simulator == 'qiskit':
        #     def expectation_function(params):
        #         return get_energy_qiskit(
        #             params, hamiltonian, noise_weights, circuit, self.config.n_qubits, 0,
        #             self.config.n_shots, self.config.phys_noise
        #         )
        elif self.config.simulator == 'qiskit':
            def expectation_function(params):
                return get_energy_qiskit_no_transpilation(
                    params, hamiltonian, noise_weights, circuit, self.config.n_qubits, 0,
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
        return self.data["solution"]
                
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
    
    def plot_UCCSD_result(self, **plot_kwargs):
        import matplotlib.pyplot as plt
        try:
            energies = np.load(f"{self.molecule}_UCCSD.npy")
        except:
            print("UCCSD data not found")
            return
        plt.scatter(self.data.index, energies, label ="UCCSD", **plot_kwargs)

    def plot_UCCSD_diff(self, **plot_kwargs):
        import matplotlib.pyplot as plt
        try:
            energies = np.load(f"{self.molecule}_UCCSD.npy")
        except:
            print("UCCSD data not found")
            return
        difference = energies - self.data["solution"]
        plt.scatter(self.data.index, difference, label ="UCCSD", **plot_kwargs)

class GroundStateEnergySavedHamiltonian(GroundStateEnergy):
    def __init__(self, config: QuantumNEATConfig, molecule: str, error_in_fitness=True, **kwargs) -> None:
        super().__init__(config, molecule, error_in_fitness, **kwargs)
        self._add_hamiltonian_to_data()

    def _add_hamiltonian_to_data(self):
        hamiltonians = []
        for _, instance in self.data.iterrows():
            hamiltonians.append(super().hamiltonian(instance))
        self.data["hamiltonian"] = hamiltonians
    
    @staticmethod
    def hamiltonian(instance):
        return instance["hamiltonian"]
    
    def energy_new(self, data):
        return self.energy(data[0], data[1])

    def gradient_new(self, data):
        return self.gradient(data[0], data[1], data[2])


if __name__ == "__main__":
    # plot_UCCSD_result("h2")
    # problem = GroundStateEnergy(None, "h2")
    # problem = GroundStateEnergySavedHamiltonian(None, "h6")
    # problem.print_hamiltonian()
    # exit()
    for molecule in ["h2", "h6", "lih"]:
        problem = GroundStateEnergySavedHamiltonian(None, molecule)
        instance = problem.data.iloc[0]
        hamiltonian = instance["hamiltonian"]
        non_zero = np.count_nonzero(hamiltonian)
        total = np.prod(np.shape(hamiltonian))
        print(f"{molecule}: {non_zero} non zero elements out of {total} elements. {non_zero/total*100:.2f}%")

    exit()
    # problem = GroundStateEnergySavedHamiltonian(None, "h2")
    problem = GroundStateEnergySavedHamiltonian(None, "h6")
    # problem = GroundStateEnergySavedHamiltonian(None, "lih")
    print(problem.data.head())
    for _, instance in problem.data.iterrows():
        # print(problem.hamiltonian(instance))
        print(np.count_nonzero(instance["hamiltonian"]))
        print(len(instance["hamiltonian"])*len(instance["hamiltonian"][0]))
        break