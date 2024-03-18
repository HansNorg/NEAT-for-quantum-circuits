from __future__ import annotations

from typing import TYPE_CHECKING

from qiskit.circuit import Parameter
from qulacs.gate import DepolarizingNoise, TwoQubitDepolarizingNoise
from quantumneat.gene import GateGene
from quantumneat.gene import GateGene
if TYPE_CHECKING:
    from quantumneat.configuration import QuantumNEATConfig, Circuit
    from quantumneat.problem import Problem

class GateCNOT(GateGene):
    n_qubits = 2

    def add_to_circuit(self, circuit:Circuit, n_parameters:int) -> tuple[Circuit, int]:
        if self.config.simulator == 'qulacs':
            circuit.add_CNOT_gate(self.qubits[0], self.qubits[1])
            if self.config.phys_noise:
                circuit.add_gate(TwoQubitDepolarizingNoise(self.qubits[0], self.qubits[1], self.config.depolarizing_noise_prob))
        elif self.config.simulator == 'qiskit':
            circuit.cnot(self.qubits[0], self.qubits[1])
            if self.config.phys_noise:
                print("Phys noise not implemented for simulator qiskit")
        else:
            raise NotImplementedError(f"Simulation method: {self.config.simulator} not implemented for {self.__class__}")
        return circuit, n_parameters
    
class GateROT(GateGene):
    n_qubits = 1
    n_parameters = 3
    
    def add_to_circuit(self, circuit:Circuit, n_parameters:int) -> tuple[Circuit, int]:
        if self.config.simulator == 'qulacs':
            circuit.add_parametric_RX_gate(self.qubits[0], self.parameters[0])
            circuit.add_parametric_RY_gate(self.qubits[0], self.parameters[1])
            circuit.add_parametric_RZ_gate(self.qubits[0], self.parameters[2])
            n_parameters += 3
            if self.config.phys_noise:
                circuit.add_gate(DepolarizingNoise(self.qubits[0], self.config.depolarizing_noise_prob))
        elif self.config.simulator == 'qiskit':
            circuit.rx(Parameter(str(n_parameters)), self.qubits[0])
            n_parameters += 1
            circuit.ry(Parameter(str(n_parameters)), self.qubits[0])
            n_parameters += 1
            circuit.rz(Parameter(str(n_parameters)), self.qubits[0])
            n_parameters += 1
            if self.config.phys_noise:
                print("Phys noise not implemented for simulator qiskit")
        else:
            raise NotImplementedError(f"Simulation method: {self.config.simulator} not implemented for {self.__class__}")
        return circuit, n_parameters

class GateRx(GateGene):
    n_qubits = 1
    n_parameters = 1
    
    def add_to_circuit(self, circuit:Circuit, n_parameters:int) -> tuple[Circuit, int]:
        if self.config.simulator == 'qulacs':
            circuit.add_parametric_RX_gate(self.qubits[0], self.parameters[0])
            n_parameters += 1
            if self.config.phys_noise:
                circuit.add_gate(DepolarizingNoise(self.qubits[0], self.config.depolarizing_noise_prob))
        elif self.config.simulator == 'qiskit':
            circuit.rx(Parameter(str(n_parameters)), self.qubits[0])
            n_parameters += 1
            if self.config.phys_noise:
                print("Phys noise not implemented for simulator qiskit")
        else:
            raise NotImplementedError(f"Simulation method: {self.config.simulator} not implemented for {self.__class__}")
        return circuit, n_parameters
    
class GateRy(GateGene):
    n_qubits = 1
    n_parameters = 1
    
    def add_to_circuit(self, circuit:Circuit, n_parameters:int) -> tuple[Circuit, int]:
        if self.config.simulator == 'qulacs':
            circuit.add_parametric_RY_gate(self.qubits[0], self.parameters[0])
            n_parameters += 1
            if self.config.phys_noise:
                circuit.add_gate(DepolarizingNoise(self.qubits[0], self.config.depolarizing_noise_prob))
        elif self.config.simulator == 'qiskit':
            circuit.ry(Parameter(str(n_parameters)), self.qubits[0])
            n_parameters += 1
            if self.config.phys_noise:
                print("Phys noise not implemented for simulator qiskit")
        else:
            raise NotImplementedError(f"Simulation method: {self.config.simulator} not implemented for {self.__class__}")
        return circuit, n_parameters
    
class GateRz(GateGene):
    n_qubits = 1
    n_parameters = 1
    
    def add_to_circuit(self, circuit:Circuit, n_parameters:int) -> tuple[Circuit, int]:
        if self.config.simulator == 'qulacs':
            circuit.add_parametric_RZ_gate(self.qubits[0], self.parameters[0])
            n_parameters += 1
            if self.config.phys_noise:
                circuit.add_gate(DepolarizingNoise(self.qubits[0], self.config.depolarizing_noise_prob))
        elif self.config.simulator == 'qiskit':
            circuit.rz(Parameter(str(n_parameters)), self.qubits[0])
            n_parameters += 1
            if self.config.phys_noise:
                print("Phys noise not implemented for simulator qiskit")
        else:
            raise NotImplementedError(f"Simulation method: {self.config.simulator} not implemented for {self.__class__}")
        return circuit, n_parameters