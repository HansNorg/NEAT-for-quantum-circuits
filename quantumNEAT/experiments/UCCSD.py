from __future__ import annotations
import os
import itertools
from typing import Union, TYPE_CHECKING
import numpy as np
from tqdm import tqdm
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qulacs import ParametricQuantumCircuit
from qulacs.gate import DepolarizingNoise, TwoQubitDepolarizingNoise

from experiments.run_experiment import cluster_n_shots
from experiments.qasm_to_qulacs import convert_QASM_to_qulacs_circuit
from quantumneat.configuration import QuantumNEATConfig
from quantumneat.problems.chemistry import GroundStateEnergy

if TYPE_CHECKING:
    from quantumneat.configuration import Circuit

class UCCSD:
    def __init__(self, config:QuantumNEATConfig, problem:GroundStateEnergy) -> None:
        self.problem = problem
        self.config = config

    def get_circuit(self) -> Union[Circuit, int]:
        n_parameters = 0
        if self.config.simulator == "qiskit":
            circuit = QuantumCircuit(QuantumRegister(self.config.n_qubits))
        elif self.config.simulator == "qulacs":
            circuit = ParametricQuantumCircuit(self.config.n_qubits)
        else:
            raise NotImplementedError(f"Simulation method: {self.config.simulator} not implemented for {self.__class__}")
        # self.problem.add_encoding_layer(circuit)
        # self.problem.add_hartree_fock_encoding(circuit)
        if self.problem.molecule == "h2":
            circuit, n_parameters = self.h2(circuit, n_parameters)
        elif self.problem.molecule == "h6":
            circuit, n_parameters = self.h6(circuit, n_parameters)
        elif self.problem.molecule == "lih":
            circuit, n_parameters = self.lih(circuit, n_parameters)
        else:
            raise NotImplementedError("UCCSD not implemented for ", self.problem.molecule)
        
        return circuit, n_parameters
        
    def ParametrizedRX(self, circuit:Circuit, qubit:int, parameter):
        if self.config.simulator == "qiskit":
            circuit.rx(parameter, qubit)
        elif self.config.simulator == "qulacs":
            circuit.add_parametric_RX_gate(qubit, parameter)
            if self.config.phys_noise:
                circuit.add_gate(DepolarizingNoise(qubit, self.config.depolarizing_noise_prob))
        else:
            raise NotImplementedError("UCCSD not implemented for ", self.problem.molecule)
        
    def RX(self, circuit:Circuit, qubit:int, angle:float):
        if self.config.simulator == "qiskit":
            circuit.rx(angle, qubit)
        elif self.config.simulator == "qulacs":
            circuit.add_RX_gate(qubit, angle)
            if self.config.phys_noise:
                circuit.add_gate(DepolarizingNoise(qubit, self.config.depolarizing_noise_prob))
        else:
            raise NotImplementedError("UCCSD not implemented for ", self.problem.molecule)
        
    def ParametrizedRY(self, circuit:Circuit, qubit:int, parameter):
        if self.config.simulator == "qiskit":
            circuit.ry(parameter, qubit)
        elif self.config.simulator == "qulacs":
            circuit.add_parametric_RY_gate(qubit, parameter)
            if self.config.phys_noise:
                circuit.add_gate(DepolarizingNoise(qubit, self.config.depolarizing_noise_prob))
        else:
            raise NotImplementedError("UCCSD not implemented for ", self.problem.molecule)
        
    def RY(self, circuit:Circuit, qubit:int, angle:float):
        if self.config.simulator == "qiskit":
            circuit.ry(angle, qubit)
        elif self.config.simulator == "qulacs":
            circuit.add_RY_gate(qubit, angle)
            if self.config.phys_noise:
                circuit.add_gate(DepolarizingNoise(qubit, self.config.depolarizing_noise_prob))
        else:
            raise NotImplementedError("UCCSD not implemented for ", self.problem.molecule)
        
    def ParametrizedRZ(self, circuit:Circuit, qubit:int, parameter):
        if self.config.simulator == "qiskit":
            circuit.rz(parameter, qubit)
        elif self.config.simulator == "qulacs":
            circuit.add_parametric_RZ_gate(qubit, parameter)
            if self.config.phys_noise:
                circuit.add_gate(DepolarizingNoise(qubit, self.config.depolarizing_noise_prob))
        else:
            raise NotImplementedError("UCCSD not implemented for ", self.problem.molecule)
        
    def RZ(self, circuit:Circuit, qubit:int, angle:float):
        if self.config.simulator == "qiskit":
            circuit.rz(angle, qubit)
        elif self.config.simulator == "qulacs":
            circuit.add_RZ_gate(qubit, angle)
            if self.config.phys_noise:
                circuit.add_gate(DepolarizingNoise(qubit, self.config.depolarizing_noise_prob))
        else:
            raise NotImplementedError("UCCSD not implemented for ", self.problem.molecule)
        
    def CNOT(self, circuit:Circuit, control:int, target:int):
        if self.config.simulator == "qiskit":
            circuit.cnot(control, target)
        elif self.config.simulator == "qulacs":
            circuit.add_CNOT_gate(control, target)
            if self.config.phys_noise:
                circuit.add_gate(TwoQubitDepolarizingNoise(control, target, self.config.depolarizing_noise_prob))
        else:
            raise NotImplementedError("UCCSD not implemented for ", self.problem.molecule)
        
    def h2(self, circuit:Circuit, n_parameters:int) -> Union[Circuit, int]:
        # n_params = 3
        # n_parameters += n_params
        # if self.config.simulator == "qiskit":
        #     parameters = [Parameter(str(ind)) for ind in range(n_params)]
        # elif self.config.simulator == "qulacs":
        #     parameters = self.config.parameter_amplitude*np.random.random(n_params)

        # self.ParametrizedRY(circuit, 0, parameters[0])
        # self.ParametrizedRY(circuit, 1, parameters[1])

        # self.RZ(circuit, 0, np.pi/2)
        # self.RY(circuit, 1, -np.pi/2)

        # self.RY(circuit, 0, np.pi/2)
        # self.RZ(circuit, 1, -np.pi)

        # self.CNOT(circuit, 1, 0)
        # self.ParametrizedRZ(circuit, 0, -parameters[2])
        # self.CNOT(circuit, 1, 0)

        # self.RX(circuit, 0, np.pi/2)
        # self.RX(circuit, 1, -np.pi/2)

        # self.CNOT(circuit, 1, 0)
        # self.ParametrizedRZ(circuit, 0, parameters[2])
        # self.CNOT(circuit, 1, 0)

        # self.RY(circuit, 0, -np.pi/2)
        # self.RY(circuit, 1, -np.pi/2)

        # self.RZ(circuit, 0, -np.pi)
        # self.RZ(circuit, 1, -np.pi/2)

        circuit = self.qiskit_ansatz("H2", 2, 2, (1,1))
        n_parameters = circuit.num_parameters
        if self.config.simulator == "qulacs":
            circuit, n_parameters = self.qiskit_to_qulacs(circuit, n_parameters)
        return circuit, n_parameters
    
    def h6(self, circuit:Circuit, n_parameters:int) -> Union[Circuit, int]:
        circuit = self.qiskit_ansatz("H6", 6, 4, (2,2))
        n_parameters = circuit.num_parameters
        if self.config.simulator == "qulacs":
            circuit, n_parameters = self.qiskit_to_qulacs(circuit, n_parameters)
        return circuit, n_parameters

    def lih(self, circuit:Circuit, n_parameters:int) -> Union[Circuit, int]:
        circuit = self.qiskit_ansatz("LiH", 8, 5, (1, 1))
        n_parameters = circuit.num_parameters
        if self.config.simulator == "qulacs":
            circuit, n_parameters = self.qiskit_to_qulacs(circuit, n_parameters)
        return circuit, n_parameters

    def qiskit_ansatz(self, molecule, n_qubits, orbitals, particles):
        from qiskit import transpile
        from qiskit.circuit import Parameter
        from qiskit.circuit.library import EfficientSU2, RXGate, RYGate, RZGate, CXGate
        from qiskit.transpiler import Target
        from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
        from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper

        target = Target(num_qubits=n_qubits)
        target.add_instruction(CXGate(), {(i,(i+1)%n_qubits):None for i in range(n_qubits)})
        target.add_instruction(RXGate(Parameter('theta')), {(i,):None for i in range(n_qubits)})
        target.add_instruction(RYGate(Parameter('theta')), {(i,):None for i in range(n_qubits)})
        target.add_instruction(RZGate(Parameter('theta')), {(i,):None for i in range(n_qubits)})
        ansatz = UCCSD(
            orbitals,
            particles,
            ParityMapper(particles),
        )
        ansatz = transpile(ansatz, target=target)
        return ansatz
    
    def qiskit_to_qulacs(self, circuit, n_parameters):
        parameters = self.config.parameter_amplitude*np.random.random(n_parameters)
        bound_circuit = circuit.bind_parameters(parameters)
        qasm = bound_circuit.qasm()
        ansatz = convert_QASM_to_qulacs_circuit(qasm.split("\n"), phys_noise=self.config.phys_noise, depol_prob=self.config.depolarizing_noise_prob)
        return ansatz, ansatz.get_parameter_count()

    def solve_problem(self, save=True, savename=""):
        circuit, n_parameters = self.get_circuit()
        parameters = self.config.parameter_amplitude*np.random.random(n_parameters)
        solutions = self.problem.evaluate(circuit, parameters)
        if save:
            os.makedirs('UCCSD', exist_ok=True)
            np.save(f"UCCSD/{self.problem}_UCCSD{savename}", solutions[1])
        return solutions
    
    def solve_problem_total(self,  save=True, savename=""):
        from quantumneat.problems.chemistry import GroundStateEnergySavedHamiltonian
        if not isinstance(self.problem, GroundStateEnergySavedHamiltonian):
            return
        circuit, n_parameters = self.get_circuit()
        parameters = self.config.parameter_amplitude*np.random.random(n_parameters)
        solutions = self.problem.evaluate_total(circuit, parameters)
        if save:
            os.makedirs('UCCSD', exist_ok=True)
            np.save(f"UCCSD/{self.problem}_UCCSD{savename}_evaluation-total", solutions[1])
        return solutions
    
def main(molecule, n_shots, args):
    if args.verbose >= 1:
        print(f"{molecule:3} {n_shots:3}", end="\r")
    savename = args.savename
    # if n_shots > 0:
    savename += f"_{n_shots}-shots"
    if args.phys_noise:
        savename += "_phys-noise"
    config = QuantumNEATConfig(n_qubits_dict[molecule], 0, n_shots=n_shots, phys_noise=args.phys_noise)
    problem = GroundStateEnergySavedHamiltonian(config, molecule)
    uccsd = UCCSD(config, problem)
    if args.verbose >=2 or args.print:
        simul = config.simulator
        config.simulator = "qiskit"
        print()
        print(uccsd.get_circuit()[0].draw())
        # print(uccsd.get_circuit()[0].draw(fold=-1))
        config.simulator = simul
        print(uccsd.get_circuit()[0])
    if args.print:
        return
    uccsd.solve_problem(savename=savename)
    uccsd.solve_problem_total(savename=savename)

if __name__ == "__main__":
    from argparse import ArgumentParser
    from quantumneat.problems.chemistry import GroundStateEnergySavedHamiltonian
    argparser = ArgumentParser()
    argparser.add_argument("molecule", nargs="+")
    argparser.add_argument("--savename", type=str, default="")
    argparser.add_argument("--n_shots", nargs="+", type=int, default=[0])
    argparser.add_argument("--phys_noise", action="store_true")
    argparser.add_argument("--shot_noise", action="store_true")
    argparser.add_argument("--print", action="store_true")
    argparser.add_argument("-v", "--verbose", action="count", default=0)
    args = argparser.parse_args()
    if args.shot_noise:
        args.n_shots = range(len(cluster_n_shots))
    n_qubits_dict = {"h2":2, "h6":6, "lih":8}
    for molecule in args.molecule:
        for shot_ind in tqdm(args.n_shots):
            n_shots = cluster_n_shots[shot_ind]
            main(molecule, n_shots, args)