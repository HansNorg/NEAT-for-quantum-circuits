import unittest
import qneat.helper as h
from qiskit import QuantumCircuit, QuantumRegister
import qneat.gate as g
import qneat.layer as l
import qneat.genome as gen
import qneat.logger as log
# from qiskit_ibm_runtime import QiskitRuntimeService
import logging

class TestHelper(unittest.TestCase):
    def setUp(self):
         self.logger = logging.getLogger("qNEAT.test")

    def test_ising_1d_instance(self):
        self.logger.info("TestHelper.test_ising_1d_instance")
        self.logger.debug(f"Ising instance: {h.ising_1d_instance(5, 0)}")
        self.logger.debug(f"Ising instance: {h.ising_1d_instance(5)}")

    def test_get_circuit_properties(self):
        self.logger.info("TestHelper.test_get_circuit_properties")
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
        self.logger.debug(str(h.get_circuit_properties(configured_circuit, ibm_backend)))
        configured_circuit, ibm_backend = h.configure_circuit_to_backend(circuit, "fake_perth")
        self.logger.debug(str(h.get_circuit_properties(configured_circuit, ibm_backend)))
        configured_circuit, ibm_backend = h.configure_circuit_to_backend(circuit, "ibmq_qasm_simulator")
        self.logger.debug(str(h.get_circuit_properties(configured_circuit, ibm_backend)))

if __name__ == '__main__':
    log.QNEATLogger("test", file_level=logging.DEBUG, mode="w")
    unittest.main()