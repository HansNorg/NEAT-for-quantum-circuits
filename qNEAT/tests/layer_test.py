import sys
sys.path.append(sys.path[0]+"\\..")
sys.path.append(sys.path[0]+"\\..\qneat")
import unittest
import qneat.layer as l
import qneat.gate as g
import qneat.logger as log
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
import logging

class TestLayer(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("qNEAT.test")

    def test_layer(self):
        self.logger.info("TestLayer.test_layer")
        register = QuantumRegister(3)
        layer = l.Layer(0)

        reference_circuit = QuantumCircuit(register)
        reference_circuit.rx(Parameter(str(0)), 0)
        reference_circuit.ry(Parameter(str(1)), 0)
        reference_circuit.rz(Parameter(str(2)), 0)
        layer.add_gate(g.GateGene(0,g.GateType.ROT,0))
        test_circuit, n_parameters = layer.add_to_circuit(QuantumCircuit(register), 0)
        self.assertEqual(str(reference_circuit), str(test_circuit))
        self.assertEqual(n_parameters, 3)

        layer.add_gate(g.GateGene(1,g.GateType.ROT,0))
        test_circuit, n_parameters = layer.add_to_circuit(QuantumCircuit(register), 0)
        self.assertEqual(str(reference_circuit), str(test_circuit))
        self.assertEqual(n_parameters, 3)

        reference_circuit2 = reference_circuit.copy()
        reference_circuit2.cnot(0, 1)
        layer.add_gate(g.GateGene(2,g.GateType.CNOT,0))
        test_circuit, n_parameters = layer.add_to_circuit(QuantumCircuit(register), 0)
        self.assertEqual(str(reference_circuit2), str(test_circuit))
        self.assertEqual(n_parameters, 3)

        layer.add_gate(g.GateGene(3,g.GateType.CNOT,0))
        test_circuit, n_parameters = layer.add_to_circuit(QuantumCircuit(register), 0)
        self.assertEqual(str(reference_circuit2), str(test_circuit))
        self.assertEqual(n_parameters, 3)

        reference_circuit.rx(Parameter(str(3)), 1)
        reference_circuit.ry(Parameter(str(4)), 1)
        reference_circuit.rz(Parameter(str(5)), 1)
        reference_circuit.cnot(0, 1)
        layer.add_gate(g.GateGene(4,g.GateType.ROT,1))
        test_circuit, n_parameters = layer.add_to_circuit(QuantumCircuit(register), 0)
        self.assertEqual(str(reference_circuit), str(test_circuit))
        
        reference_circuit.cnot(1, 2)
        layer.add_gate(g.GateGene(0,g.GateType.CNOT,1))
        test_circuit, n_parameters = layer.add_to_circuit(QuantumCircuit(register), 0)
        self.assertEqual(str(reference_circuit), str(test_circuit))

        reference_circuit.cnot(2, 0)
        layer.add_gate(g.GateGene(6,g.GateType.CNOT,2))
        test_circuit, n_parameters = layer.add_to_circuit(QuantumCircuit(register), 0)
        self.assertEqual(str(reference_circuit), str(test_circuit))

        # print(test_circuit)

if __name__ == '__main__':
    log.QNEATLogger("test", mode="w")
    unittest.main()