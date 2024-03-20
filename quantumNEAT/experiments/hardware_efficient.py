from __future__ import annotations
import itertools
from typing import Union, TYPE_CHECKING
import numpy as np
from tqdm import tqdm
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qulacs import ParametricQuantumCircuit
from qulacs.gate import DepolarizingNoise, TwoQubitDepolarizingNoise

from experiments.run_experiment import cluster_n_shots
from quantumneat.configuration import QuantumNEATConfig
from quantumneat.problem import Problem

if TYPE_CHECKING:
    from quantumneat.configuration import Circuit

class HardwareEfficient:
    def __init__(self, config:QuantumNEATConfig, problem:Problem) -> None:
        self.problem = problem
        self.config = config

    def get_circuit(self, layers:int) -> Union[Circuit, int]:
        n_parameters = 0
        if self.config.simulator == "qiskit":
            circuit = QuantumCircuit(QuantumRegister(self.config.n_qubits))
            for layer in range(layers):
                circuit, n_parameters = self.add_R_layer_qiskit(circuit, n_parameters)
                circuit, n_parameters = self.add_CNOT_layer_qiskit(circuit, n_parameters)
            circuit, n_parameters = self.add_R_layer_qiskit(circuit, n_parameters)
        elif self.config.simulator == "qulacs":
            circuit = ParametricQuantumCircuit(self.config.n_qubits)
            for layer in range(layers):
                circuit, n_parameters = self.add_R_layer_qulacs(circuit, n_parameters)
                circuit, n_parameters = self.add_CNOT_layer_qulacs(circuit, n_parameters)
            circuit, n_parameters = self.add_R_layer_qulacs(circuit, n_parameters)
        else:
            raise NotImplementedError(f"Simulation method: {self.config.simulator} not implemented for {self.__class__}")
        return circuit, n_parameters
        
    def add_R_layer_qulacs(self, circuit:ParametricQuantumCircuit, n_parameters:int) -> Union[ParametricQuantumCircuit, int]:
        for qubit in range(self.config.n_qubits):
            circuit.add_parametric_RY_gate(qubit, 0)
            circuit.add_parametric_RZ_gate(qubit, 0)
            n_parameters += 2
            if self.config.phys_noise:
                circuit.add_gate(DepolarizingNoise(qubit, self.config.depolarizing_noise_prob))
        return circuit, n_parameters

    def add_CNOT_layer_qulacs(self, circuit:ParametricQuantumCircuit, n_parameters:int) -> Union[ParametricQuantumCircuit, int]:
        for qubit in range(0, self.config.n_qubits, 2):
            circuit.add_CNOT_gate(qubit, (qubit+1)%self.config.n_qubits)
            if self.config.phys_noise:
                circuit.add_gate(TwoQubitDepolarizingNoise(qubit, (qubit+1)%self.config.n_qubits, self.config.depolarizing_noise_prob))
        for qubit in range(1, self.config.n_qubits, 2):
            circuit.add_CNOT_gate(qubit, (qubit+1)%self.config.n_qubits)
            if self.config.phys_noise:
                circuit.add_gate(TwoQubitDepolarizingNoise(qubit, (qubit+1)%self.config.n_qubits, self.config.depolarizing_noise_prob))
        return circuit, n_parameters
    
    def add_R_layer_qiskit(self, circuit:QuantumCircuit, n_parameters:int) -> Union[QuantumCircuit, int]:
        for qubit in range(self.config.n_qubits):
            circuit.ry(Parameter(str(n_parameters)), qubit)
            circuit.rz(Parameter(str(n_parameters+1)), qubit)
            n_parameters += 2
        return circuit, n_parameters

    def add_CNOT_layer_qiskit(self, circuit:QuantumCircuit, n_parameters:int) -> Union[QuantumCircuit, int]:
        for qubit in range(0, self.config.n_qubits, 2):
            circuit.cnot(qubit, (qubit+1)%self.config.n_qubits)
        for qubit in range(1, self.config.n_qubits, 2):
            circuit.cnot(qubit, (qubit+1)%self.config.n_qubits)
        return circuit, n_parameters
    
    def solve_problem(self, layers:int, save=True, savename=""):
        circuit, n_parameters = self.get_circuit(layers)
        parameters = self.config.parameter_amplitude*np.random.random(n_parameters)
        solutions = self.problem.evaluate(circuit, parameters)
        if save:
            np.save(f"{self.problem}_HE_{layers}-layers{savename}", solutions[1])
        return solutions
    
    def solve_problem_total(self, layers:int, save=True, savename=""):
        from quantumneat.problems.chemistry import GroundStateEnergySavedHamiltonian
        if not isinstance(self.problem, GroundStateEnergySavedHamiltonian):
            return
        circuit, n_parameters = self.get_circuit(layers)
        parameters = self.config.parameter_amplitude*np.random.random(n_parameters)
        solutions = self.problem.evaluate_total(circuit, parameters)
        if save:
            np.save(f"{self.problem}_HE_{layers}-layers{savename}_evaluation-total", solutions[1])
        return solutions
    
def main(molecule, layers, n_shots, args):
    savename = args.savename
    if n_shots > 0:
        savename += f"_{n_shots}-shots"
    if args.phys_noise:
        savename += "_phys-noise"
    config = QuantumNEATConfig(n_qubits_dict[molecule], 0, n_shots=n_shots, phys_noise=args.phys_noise)
    problem = GroundStateEnergySavedHamiltonian(config, molecule)
    he = HardwareEfficient(config, problem)
    for layers in tqdm(layers, disable=args.batch_job):
        if args.verbose >=2:
            simul = config.simulator
            config.simulator = "qiskit"
            print(he.get_circuit(layers)[0].draw(fold=-1))
            config.simulator = simul
        if args.verbose >= 1:
            print(f"{molecule:3} {layers:2} layers", end="\r")
        he.solve_problem(layers, savename=savename)
        he.solve_problem_total(layers, savename=savename)

if __name__ == "__main__":
    from argparse import ArgumentParser
    from quantumneat.problems.chemistry import GroundStateEnergySavedHamiltonian
    argparser = ArgumentParser()
    argparser.add_argument("molecule", nargs="+")
    argparser.add_argument("--savename", type=str, default="")
    argparser.add_argument("--n_shots", nargs="+", type=int, default=[0])
    argparser.add_argument("--phys_noise", action="store_true")
    argparser.add_argument("--shot_noise", action="store_true")
    argparser.add_argument("--layers", nargs="+", type=int, default=[0, 1, 2, 4, 8, 16])
    argparser.add_argument("--batch_job", action="store_true")
    argparser.add_argument("-v", "--verbose", action="count", default=0)
    args = argparser.parse_args()
    if args.shot_noise:
        args.n_shots = range(len(cluster_n_shots))
    n_qubits_dict = {"h2":2, "h6":6, "lih":8}
    if args.batch_job:
        batch_tasks = list(itertools.product(["h2", "h6", "lih"], [0, 1, 2, 4, 8, 16]))
        for task_ind in args.molecule:
            for shot_ind in args.n_shots:
                n_shots = cluster_n_shots[shot_ind]
                task = batch_tasks[int(task_ind)]
                main(task[0], [task[1]], n_shots, args)
    else:
        for molecule in args.molecule:
            for shot_ind in args.n_shots:
                n_shots = cluster_n_shots[shot_ind]
                main(molecule, args.layers, n_shots, args)