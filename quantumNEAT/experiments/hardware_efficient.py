from typing import Union
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qulacs import ParametricQuantumCircuit

from quantumneat.configuration import QuantumNEATConfig
from quantumneat.problem import Problem

class HardwareEfficient:
    def __init__(self, config:QuantumNEATConfig, problem:Problem) -> None:
        self.problem = problem
        self.config = config

    def get_circuit(self, layers:int):
        n_parameters = 0
        if self.config.simulator == "qiskit":
            circuit = QuantumCircuit(QuantumRegister(self.config.n_qubits))
            for layer in range(layers):
                circuit, n_parameters = self.add_layer_qiskit(circuit, n_parameters)
        elif self.config.simulator == "qulacs":
            circuit = ParametricQuantumCircuit(self.config.n_qubits)
            for layer in range(layers):
                circuit, n_parameters = self.add_layer_qulacs(circuit, n_parameters)
        else:
            raise NotImplementedError(f"Simulation method: {self.config.simulator} not implemented for {self.__class__}")
        return circuit, n_parameters
        
    def add_layer_qulacs(self, circuit:ParametricQuantumCircuit, n_parameters:int) -> Union[ParametricQuantumCircuit, int]:
        for qubit in range(self.config.n_qubits):
            circuit.add_parametric_RY_gate(qubit, 0)
            circuit.add_parametric_RZ_gate(qubit, 0)
            n_parameters += 2
        for qubit in range(self.config.n_qubits):
            circuit.add_CNOT_gate(qubit, (qubit+1)%self.config.n_qubits)
        return circuit, n_parameters
    
    def add_layer_qiskit(self, circuit:QuantumCircuit, n_parameters:int) -> Union[QuantumCircuit, int]:
        for qubit in range(self.config.n_qubits):
            circuit.ry(Parameter(str(n_parameters)), qubit)
            circuit.rz(Parameter(str(n_parameters+1)), qubit)
            n_parameters += 2
        for qubit in range(self.config.n_qubits):
            circuit.cnot(qubit, (qubit+1)%self.config.n_qubits)
        return circuit, n_parameters
    
    def solve_problem(self, layers:int, save=True, savename=""):
        circuit, n_parameters = self.get_circuit(layers)
        parameters = self.config.parameter_amplitude*np.random.random(n_parameters)
        solutions = self.problem.evaluate(circuit, parameters)
        if save:
            np.save(f"{self.problem}_HE_{layers}-layers{savename}", solutions[1])
        return solutions
    
if __name__ == "__main__":
    from argparse import ArgumentParser
    from quantumneat.problems.chemistry import GroundStateEnergySavedHamiltonian
    argparser = ArgumentParser()
    argparser.add_argument("molecule", nargs="+")
    argparser.add_argument("--savename", type=str, default="")
    argparser.add_argument("-v", "--verbose", action="count", default=0)
    args = argparser.parse_args()

    n_qubits_dict = {"h2":2, "h6":6, "lih":8}
    for molecule in args.molecule:
        config = QuantumNEATConfig(n_qubits_dict[molecule], 0)
        problem = GroundStateEnergySavedHamiltonian(config, molecule)
        he = HardwareEfficient(config, problem)
        for layers in [1, 2, 4, 8, 16]:
            if args.verbose >= 1:
                print(f"{molecule:3} {layers:2} layers", end="\r")
            he.solve_problem(layers, savename=args.savename)