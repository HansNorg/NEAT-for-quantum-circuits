import sys
sys.path.append(sys.path[0]+"\\..")
import unittest
import qneat.gate as g
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

class TestGate(unittest.TestCase):
    def test_ROTgate(self):
        n_qubits = 1
        register = QuantumRegister(n_qubits)
        test_circuit = QuantumCircuit(register)
        reference_circuit = QuantumCircuit(register)

        gate = g.GateGene(0, g.GateType.ROT, 0)
        test_circuit, n_parameters = gate.add_to_circuit(test_circuit, 0)
        reference_circuit.rx(Parameter(str(0)), 0)
        reference_circuit.ry(Parameter(str(1)), 0)
        reference_circuit.rz(Parameter(str(2)), 0)
        self.assertEqual(str(test_circuit), str(reference_circuit), "Different circuits")
        self.assertEqual(n_parameters, 3, "Wrong number of parameters")


        n_qubits = 5
        register = QuantumRegister(n_qubits)
        test_circuit = QuantumCircuit(register)
        reference_circuit = QuantumCircuit(register)

        gate1 = g.GateGene(0, g.GateType.ROT, 2)
        gate2 = g.GateGene(3, g.GateType.ROT, 4)
        test_circuit, n_parameters = gate1.add_to_circuit(test_circuit, 0)
        test_circuit, n_parameters = gate2.add_to_circuit(test_circuit, n_parameters)
        reference_circuit.rx(Parameter(str(0)), 2)
        reference_circuit.ry(Parameter(str(1)), 2)
        reference_circuit.rz(Parameter(str(2)), 2)
        reference_circuit.rx(Parameter(str(3)), 4)
        reference_circuit.ry(Parameter(str(4)), 4)
        reference_circuit.rz(Parameter(str(5)), 4)
        self.assertEqual(str(test_circuit), str(reference_circuit), "Different circuits")
        self.assertEqual(n_parameters, 6, "Wrong number of parameters")

    def test_CNOTgate_error(self):
        circuit = QuantumCircuit(QuantumRegister(1))
        gate = g.GateGene(0, g.GateType.CNOT, 0)
        self.assertRaises(ValueError, gate.add_to_circuit, circuit, 0)

    def test_CNOTgate(self):
        n_qubits = 2
        register = QuantumRegister(n_qubits)
        test_circuit = QuantumCircuit(register)
        reference_circuit = QuantumCircuit(register)
        gate = g.GateGene(0, g.GateType.CNOT, 0)

        test_circuit, n_parameters = gate.add_to_circuit(test_circuit, 0)
        reference_circuit.cnot(0, 1)
        self.assertEqual(str(test_circuit), str(reference_circuit), "Different circuits")
        self.assertEqual(n_parameters, 0, "Wrong number of parameters")

        n_qubits = 5
        register = QuantumRegister(n_qubits)
        test_circuit = QuantumCircuit(register)
        reference_circuit = QuantumCircuit(register)
        gate = g.GateGene(0, g.GateType.CNOT, 3)

        test_circuit, n_parameters = gate.add_to_circuit(test_circuit, 0)
        reference_circuit.cnot(3, 4)
        self.assertEqual(str(test_circuit), str(reference_circuit), "Different circuits")
        self.assertEqual(n_parameters, 0, "Wrong number of parameters")

        n_qubits = 5
        register = QuantumRegister(n_qubits)
        test_circuit = QuantumCircuit(register)
        reference_circuit = QuantumCircuit(register)
        gate = g.GateGene(0, g.GateType.CNOT, 4)

        test_circuit, n_parameters = gate.add_to_circuit(test_circuit, 0)
        reference_circuit.cnot(4, 0)
        self.assertEqual(str(test_circuit), str(reference_circuit), "Different circuits")
        self.assertEqual(n_parameters, 0, "Wrong number of parameters")

    def test_get_distance_ROT(self):
        gate1 = g.GateGene(0, g.GateType.ROT, 0)
        gate2 = g.GateGene(0, g.GateType.ROT, 0)

        self.assertEqual(g.GateGene.get_distance(gate1, gate1), (True, 0), "Distance to itself should be 0")
        self.assertEqual(g.GateGene.get_distance(gate2, gate2), (True, 0), "Distance to itself should be 0")

        gate1.parameters = [0, 0, 0]
        gate2.parameters = [1, 2, 2]
        self.assertEqual(g.GateGene.get_distance(gate1, gate2), (True, 3), "Preset parameter distance check")

        gate1.parameters = [5, 6, 7]
        gate2.parameters = [5, 6, 5]
        self.assertEqual(g.GateGene.get_distance(gate1, gate2), (True, 2), "Preset parameter distance check")

    def test_get_distance_CNOT(self):
        gate1 = g.GateGene(0, g.GateType.CNOT, 0)
        gate2 = g.GateGene(0, g.GateType.CNOT, 0)
        self.assertEqual(g.GateGene.get_distance(gate1, gate1), (False, 0), "Distance to itself should be 0")
        self.assertEqual(g.GateGene.get_distance(gate2, gate2), (False, 0), "Distance to itself should be 0")
        self.assertEqual(g.GateGene.get_distance(gate1, gate2), (False, 0), "Distance between parameterless gates should be 0")
        
    def test_get_distance_error(self):
        gate1 = g.GateGene(0, g.GateType.ROT, 0)
        gate2 = g.GateGene(0, g.GateType.CNOT, 0)
        self.assertRaises(ValueError, g.GateGene.get_distance, gate1, gate2)

if __name__ == '__main__':
    unittest.main()