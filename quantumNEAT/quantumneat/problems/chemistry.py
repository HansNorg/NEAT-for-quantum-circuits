from __future__ import annotations

from time import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
# from qiskit_nature.second_q.circuit.library import HartreeFock
# from qiskit_nature.second_q.mappers import ParityMapper
from qulacs.gate import DepolarizingNoise
from scipy.optimize import minimize

from quantumneat.configuration import QuantumNEATConfig
from quantumneat.helper import get_energy_qulacs, get_energy_qiskit, get_energy_qiskit_no_transpilation
from quantumneat.problem import Problem
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
            if key == "solution" or key == "correction" or key == "hamiltonian" or key == "weights":
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

    def add_hartree_fock_encoding(self, circuit:Circuit):
        if self.molecule == "h2":
            if self.config.simulator == "qulacs":
                circuit.add_X_gate(0)
                if self.config.phys_noise_encoding:
                    circuit.add_gate(DepolarizingNoise(0, self.config.depolarizing_noise_prob))
            elif self.config.simulator == "qiskit":
                circuit.x(0)
                if self.config.phys_noise_encoding:
                    print("Phys noise not implemented for simulator qiskit")
            else:
                raise NotImplementedError(f"Simulator {self.config.simulator} not implemented for add_hartree_fock_encoding")
        elif self.molecule == "h6":
            if self.config.simulator == "qulacs":
                circuit.add_X_gate(0)
                circuit.add_X_gate(3)
                if self.config.phys_noise_encoding:
                    circuit.add_gate(DepolarizingNoise(0, self.config.depolarizing_noise_prob))
                    circuit.add_gate(DepolarizingNoise(3, self.config.depolarizing_noise_prob))
            elif self.config.simulator == "qiskit":
                circuit.x(0)
                circuit.x(3)
                if self.config.phys_noise_encoding:
                    print("Phys noise not implemented for simulator qiskit")
            else:
                raise NotImplementedError(f"Simulator {self.config.simulator} not implemented for add_hartree_fock_encoding")
        elif self.molecule == "lih":
            if self.config.simulator == "qulacs":
                for i in range(0, 4):
                    circuit.add_X_gate(i)             
                if self.config.phys_noise_encoding:
                    for i in range(0, 4):
                        circuit.add_gate(DepolarizingNoise(i, self.config.depolarizing_noise_prob))
            elif self.config.simulator == "qiskit":
                for i in range(0, 4):
                    circuit.x(i)
                if self.config.phys_noise_encoding:
                    print("Phys noise not implemented for simulator qiskit")
            else:
                raise NotImplementedError(f"Simulator {self.config.simulator} not implemented for add_hartree_fock_encoding")
        else:
            raise NotImplementedError(f"add_hartree_fock_encoding not implemented for {self.molecule}")

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
        if self.config.use_total_energy:
            energy = sum(self.total_energy(circuit, parameters, no_optimization))
            if subtract_solution:
                energy -= sum([instance["solution"] for _, instance in self.data.iterrows()])
            return energy
        energy = 0
        for _, instance in self.data.iterrows():
            energy += self.instance_energy(instance, circuit, parameters, no_optimization)
            if subtract_solution:
                energy -= instance["solution"]
        return energy

    def instance_energy(self, instance, circuit, parameters, no_optimization = False):
        hamiltonian = self.hamiltonian(instance)
        correction = instance.loc["correction"]
        # noise_weights = np.ones(self.config.n_qubits)
        noise_weights = self.noise_weights(instance)
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
    
    def total_energy(self, circuit, parameters, no_optimization = False):
        if self.config.simulator == "qulacs":
            noise_weigths = np.ones(self.config.n_qubits)
            def expectation_function(params):
                energy = 0
                for _, instance in self.data.iterrows():
                    hamiltonian = self.hamiltonian(instance)
                    correction = instance.loc["correction"]
                    energy += get_energy_qulacs(params, hamiltonian, noise_weigths, circuit,
                                                self.config.n_qubits, correction, self.config.n_shots,
                                                self.config.phys_noise)
                return energy
        else:
            raise NotImplementedError(f"Simulator {self.config.simulator} not implemented for total_energy of GroundStateEnergy")
        if self.config.optimize_energy and not no_optimization:
            optimized_parameters = minimize(expectation_function, parameters, method="COBYLA", tol=1e-4,
                                   options={'maxiter':self.config.optimize_energy_max_iter}
                                   ).x
        else:
            optimized_parameters = parameters
        energies = []
        for _, instance in self.data.iterrows():
            energies.append(self.instance_energy(instance, circuit, optimized_parameters, no_optimization=True))
        return energies

    @staticmethod
    def hamiltonian(instance:pd.DataFrame) -> list:
        H = 0
        for string, const in instance.items():
            if string == "correction" or string == "solution":
                continue
            H += from_string(string)*const
        return H

    @staticmethod
    def noise_weights(instance:pd.DataFrame) -> list:
        weights = []
        for string, const in instance.items():
            if string == "correction" or string == "solution" or string == "hamiltonian":
                continue
            weights.append(const.real)
        return weights
    
    def solution(self) -> float:
        return self.data["solution"]
                
    def evaluate(self, circuit:Circuit, parameters, N = 1000):
        max_iter = self.config.optimize_energy_max_iter
        optimize_energy = self.config.optimize_energy
        self.config.optimize_energy = self.config.optimize_energy_evaluation
        self.config.optimize_energy_max_iter = N
        
        energies = []
        for _, instance in self.data.iterrows():
            energies.append(self.instance_energy(instance, circuit, parameters))

        self.config.optimize_energy_max_iter = max_iter
        self.config.optimize_energy = optimize_energy
        return self.data.index, energies
    
    def evaluate_total(self, circuit:Circuit, parameters, N = 1000):
        max_iter = self.config.optimize_energy_max_iter
        optimize_energy = self.config.optimize_energy
        self.config.optimize_energy = self.config.optimize_energy_evaluation
        self.config.optimize_energy_max_iter = N
        
        # energies = []
        # for _, instance in self.data.iterrows():
        #     energies.append(self.instance_energy(instance, circuit, parameters))
        energies = self.total_energy(circuit, parameters)

        self.config.optimize_energy_max_iter = max_iter
        self.config.optimize_energy = optimize_energy
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

    def plot_UCCSD_diff(self, absolute = False, **plot_kwargs):
        import matplotlib.pyplot as plt
        try:
            energies = np.load(f"{self.molecule}_UCCSD.npy")
        except:
            print("UCCSD data not found")
            return
        difference = energies - self.data["solution"]
        if absolute:
            difference = abs(difference)
        plt.scatter(self.data.index, difference, label ="UCCSD", **plot_kwargs)

    def plot_HE_result(self, layers, n_shots=-1, phys_noise=False, **plot_kwargs):
        import matplotlib.pyplot as plt
        extra = ""
        if n_shots != -1:
            extra += f"_{n_shots}-shots"
        if phys_noise:
            extra += "_phys-noise"
        try:
            energies = np.load(f"{self.molecule}_HE_{layers}-layers{extra}.npy")
        except:
            print(f"HE data not found for {self.molecule}{extra} {layers} layers")
            return
        plt.scatter(self.data.index, energies, label =f"HE-{layers}{extra}", **plot_kwargs)

    def plot_HE_result_total(self, layers, n_shots=-1, phys_noise=False, **plot_kwargs):
        import matplotlib.pyplot as plt
        extra = ""
        if n_shots != -1:
            extra += f"_{n_shots}-shots"
        if phys_noise:
            extra += "_phys-noise"
        try:
            energies = np.load(f"{self.molecule}_HE_{layers}-layers{extra}_evaluation-total.npy")
        except:
            print(f"HE data not found for {self.molecule}{extra} {layers} layers")
            return
        plt.scatter(self.data.index, energies, label =f"HE-{layers}{extra}", **plot_kwargs)

    def plot_HE_diff(self, layers, n_shots=-1, phys_noise=False, absolute = False, **plot_kwargs):
        import matplotlib.pyplot as plt
        extra = ""
        if n_shots != -1:
            extra += f"_{n_shots}-shots"
        if phys_noise:
            extra += "_phys-noise"
        try:
            energies = np.load(f"{self.molecule}_HE_{layers}-layers{extra}.npy")
        except:
            print(f"HE data not found for {self.molecule}{extra} {layers} layers")
            return
        difference = energies - self.data["solution"]
        if absolute:
            difference = abs(difference)
        plt.scatter(self.data.index, difference, label =f"HE-{layers}{extra}", **plot_kwargs)

    def plot_HE_diff_total(self, layers, n_shots=-1, phys_noise=False, absolute=False, **plot_kwargs):
        import matplotlib.pyplot as plt
        extra = ""
        if n_shots != -1:
            extra += f"_{n_shots}-shots"
        if phys_noise:
            extra += "_phys-noise"
        try:
            energies = np.load(f"{self.molecule}_HE_{layers}-layers{extra}_evaluation-total.npy")
        except:
            print(f"HE data not found for {self.molecule}{extra} {layers} layers")
            return
        difference = energies - self.data["solution"]
        if absolute:
            difference = abs(difference)
        plt.scatter(self.data.index, difference, label =f"HE-{layers}{extra}", **plot_kwargs)

    def plot_adaptVQE_result(self, **plot_kwargs):
        import matplotlib.pyplot as plt
        try:
            energies = np.load(f"{self.molecule}_AdaptVQE.npy")
        except:
            print(f"AdaptVQE data not found")
            return
        plt.scatter(self.data.index, energies, label =f"AdaptVQE", **plot_kwargs)

    def plot_adaptVQE_diff(self, absolute=False, **plot_kwargs):
        import matplotlib.pyplot as plt
        try:
            energies = np.load(f"{self.molecule}_AdaptVQE.npy")
        except:
            print(f"AdaptVQE data not found")
            return
        difference = energies - self.data["solution"]
        if absolute:
            difference = abs(difference)
        plt.scatter(self.data.index, difference, label =f"AdaptVQE", **plot_kwargs)

    def __str__(self) -> str:
        return self.molecule

class GroundStateEnergySavedHamiltonian(GroundStateEnergy):
    def __init__(self, config: QuantumNEATConfig, molecule: str, error_in_fitness=True, **kwargs) -> None:
        super().__init__(config, molecule, error_in_fitness, **kwargs)
        self._add_hamiltonian_to_data()
        self._add_weights_to_data()

    def _add_hamiltonian_to_data(self):
        hamiltonians = []
        for _, instance in self.data.iterrows():
            hamiltonians.append(super().hamiltonian(instance))
        self.data["hamiltonian"] = hamiltonians

    def _add_weights_to_data(self):
        weights = []
        for _, instance in self.data.iterrows():
            weights.append(super().noise_weights(instance))
        self.data["weights"] = weights
    
    @staticmethod
    def hamiltonian(instance):
        return instance["hamiltonian"]
    
    @staticmethod
    def noise_weights(instance):
        return instance["weights"]
    
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