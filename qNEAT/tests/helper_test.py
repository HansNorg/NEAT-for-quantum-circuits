import sys
sys.path.append(sys.path[0]+"\\..")
sys.path.append(sys.path[0]+"\\..\qneat")
import unittest
import qneat.helper as h
from qiskit import QuantumCircuit, QuantumRegister
import qneat.gate as g
import qneat.layer as l
import qneat.genome as gen
from qiskit_ibm_runtime import QiskitRuntimeService

class TestHelper(unittest.TestCase):
    def test_ising_1d_instance(self):
        if __name__ == '__main__':
            print(h.ising_1d_instance(5, 0))
            print(h.ising_1d_instance(5))

    def test_get_circuit_properties(self):
        if __name__ != '__main__':
            return
        global_layer_number = h.GlobalLayerNumber()
        layers = {0: l.Layer(0), 1:l.Layer(1)}
        global_layer_number._layer_number = 2

        layers[0].add_gate(g.GateGene(0, g.GateType.ROT, 0))
        layers[0].add_gate(g.GateGene(1, g.GateType.ROT, 1))
        layers[0].add_gate(g.GateGene(2, g.GateType.CNOT, 0))
        layers[0].add_gate(g.GateGene(3, g.GateType.CNOT, 1))
        
        genome = gen.Genome.from_layers(global_layer_number, layers)
        circuit, n_parameters = genome.get_circuit(3)
        configured_circuit, ibm_backend = h.configure_circuit_to_backend(circuit, "ibm_perth")
        print(h.get_circuit_properties(configured_circuit, ibm_backend))
        configured_circuit, ibm_backend = h.configure_circuit_to_backend(circuit, "fake_perth")
        print(h.get_circuit_properties(configured_circuit, ibm_backend))
        configured_circuit, ibm_backend = h.configure_circuit_to_backend(circuit, "ibmq_qasm_simulator")
        print(h.get_circuit_properties(configured_circuit, ibm_backend))


if __name__ == '__main__':
    unittest.main()