import sys
sys.path.append(sys.path[0]+"\\..")
sys.path.append(sys.path[0]+"\\..\qneat")
import qneat.genome as gen
import qneat.gate as g
import qneat.layer as l
import qneat.helper as h
import qneat.logger as log
import unittest
from qiskit import QuantumCircuit, QuantumRegister
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# import warnings filter
from warnings import simplefilter
import logging

class TestGenome(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("qNEAT.test")
        self.logger.info("TestGenome.setUp")
        global_layer_number = h.GlobalLayerNumber()
        self.genome = gen.Genome(global_layer_number)
        self.gate1 = g.GateGene(0, g.GateType.ROT, 0)
        self.gate2 = g.GateGene(1, g.GateType.ROT, 1)
        self.gate3 = g.GateGene(2, g.GateType.CNOT, 0)
        self.gate4 = g.GateGene(3, g.GateType.CNOT, 1)
        self.gate5 = g.GateGene(4, g.GateType.ROT, 0)
        self.gate6 = g.GateGene(5, g.GateType.CNOT, 1)

        global_layer_number = h.GlobalLayerNumber()
        self.layers1 = {0: l.Layer(0), 1:l.Layer(1), 2:l.Layer(2)}
        self.layers2 = {0: l.Layer(0), 1:l.Layer(1), 2:l.Layer(2)}
        global_layer_number._layer_number = 3

        self.genome1 = gen.Genome.from_layers(global_layer_number, self.layers1)
        self.genome2 = gen.Genome.from_layers(global_layer_number, self.layers2)

    def test_genome(self):
        self.logger.info("TestGenome.test_genome")
        self.assertTrue(self.genome.add_gate(self.gate1))
        self.assertTrue(self.genome.add_gate(self.gate2))
        self.assertTrue(self.genome.add_gate(self.gate3))
        self.assertTrue(self.genome.add_gate(self.gate4))

        if __name__ == '__main__':
            tries = 0
            while not self.genome.add_gate(self.gate1):
                tries += 1
                if tries > 100:
                    break
            while not self.genome.add_gate(self.gate2):
                tries += 1
                if tries > 100:
                    break
            while not self.genome.add_gate(self.gate3):
                tries += 1
                if tries > 100:
                    break
            while not self.genome.add_gate(self.gate4):
                tries += 1
                if tries > 100:
                    break
            print(self.genome.get_circuit(2)[0])
            print(f"tries: {tries}")

    def test_line_up(self):
        self.logger.info("TestGenome.test_line_up")
        self.layers1[0].add_gate(self.gate1)
        self.assertEqual(gen.Genome.line_up(self.genome1, self.genome2), (0, 1, 0, 0))
        self.layers2[0].add_gate(self.gate1)
        self.assertEqual(gen.Genome.line_up(self.genome1, self.genome2), (1, 0, 0, 0))
        self.layers2[0].add_gate(self.gate2)
        self.assertEqual(gen.Genome.line_up(self.genome1, self.genome2), (1, 1, 0, 0))
        self.layers1[0].add_gate(self.gate3)
        self.assertEqual(gen.Genome.line_up(self.genome1, self.genome2), (1, 2, 0, 0))
        self.layers2[0].add_gate(self.gate3)
        self.assertEqual(gen.Genome.line_up(self.genome1, self.genome2), (2, 1, 0, 0))
        self.layers2[0].add_gate(self.gate4)
        self.assertEqual(gen.Genome.line_up(self.genome1, self.genome2), (2, 2, 0, 0))
        self.layers1[2].add_gate(self.gate2)
        self.assertEqual(gen.Genome.line_up(self.genome1, self.genome2), (2, 3, 0, 0))
        self.layers1[2].add_gate(self.gate4)
        self.assertEqual(gen.Genome.line_up(self.genome1, self.genome2), (2, 4, 0, 0))
        self.layers2[2].add_gate(self.gate2)
        self.assertEqual(gen.Genome.line_up(self.genome1, self.genome2), (3, 3, 0, 0))
        self.layers2[2].add_gate(self.gate4)
        self.assertEqual(gen.Genome.line_up(self.genome1, self.genome2), (4, 2, 0, 0))

        distance = g.GateGene.get_distance(self.gate1, self.gate5)[1]/3
        self.layers1[1].add_gate(self.gate1)
        self.layers2[1].add_gate(self.gate5)
        self.assertEqual(gen.Genome.line_up(self.genome1, self.genome2), (5, 2, 0, distance))
        self.layers1[1].add_gate(self.gate4)
        self.assertEqual(gen.Genome.line_up(self.genome1, self.genome2), (5, 3, 0, distance))
        self.layers2[1].add_gate(self.gate6)
        self.assertEqual(gen.Genome.line_up(self.genome1, self.genome2), (6, 2, 0, distance))

    def test_compatibility_distance(self):
        self.logger.info("TestGenome.test_compatibility_distance")
        self.layers1[0].add_gate(self.gate1)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 1, 1, 1), 1)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 3, 3, 4), 3)
        self.layers2[0].add_gate(self.gate1)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 1, 1, 1), 0)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 3, 3, 4), 0)
        self.layers2[0].add_gate(self.gate2)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 1, 1, 1), .5)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 3, 3, 4), 1.5)
        self.layers1[0].add_gate(self.gate3)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 1, 1, 1), 1)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 3, 3, 4), 3)
        self.layers2[0].add_gate(self.gate3)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 1, 1, 1), 1/3)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 3, 3, 4), 1)
        self.layers2[0].add_gate(self.gate4)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 1, 1, 1), 2/4)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 3, 3, 4), 6/4)
        self.layers1[2].add_gate(self.gate2)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 1, 1, 1), 3/4)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 3, 3, 4), 9/4)
        self.layers1[2].add_gate(self.gate4)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 1, 1, 1), 4/4)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 3, 3, 4), 12/4)
        self.layers2[2].add_gate(self.gate2)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 1, 1, 1), 3/5)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 3, 3, 4), 9/5)
        self.layers2[2].add_gate(self.gate4)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 1, 1, 1), 2/6)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 3, 3, 4), 6/6)

        distance = g.GateGene.get_distance(self.gate1, self.gate5)[1]/3
        self.layers1[1].add_gate(self.gate1)
        self.layers2[1].add_gate(self.gate5)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 1, 1, 1), 2/7+distance)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 3, 3, 4), 6/7+4*distance)
        self.layers1[1].add_gate(self.gate4)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 1, 1, 1), 3/7+distance)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 3, 3, 4), 9/7+4*distance)
        self.layers2[1].add_gate(self.gate6)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 1, 1, 1), 2/8+distance)
        self.assertEqual(gen.Genome.compatibility_distance(self.genome1, self.genome2, 3, 3, 4), 6/8+4*distance)

        #TODO add/update tests to account for difference between excess and disjoint

    def test_compute_gradient(self):
        self.logger.info("TestGenome.test_compute_gradient")
        self.assertEqual(self.genome1.compute_gradient(self.genome1.get_circuit(3)[0],0),0)
        self.layers1[0].add_gate(self.gate3)
        self.assertEqual(self.genome1.compute_gradient(self.genome1.get_circuit(3)[0],0),0)
        self.layers1[0].add_gate(self.gate1)
        self.assertNotEqual(self.genome1.compute_gradient(self.genome1.get_circuit(3)[0],3), 0)
        self.layers1[0].add_gate(self.gate2)
        self.assertNotEqual(self.genome1.compute_gradient(self.genome1.get_circuit(3)[0],6), 0)
        self.layers1[0].add_gate(self.gate4)
        self.assertNotEqual(self.genome1.compute_gradient(self.genome1.get_circuit(3)[0],6), 0)
        self.layers1[2].add_gate(self.gate1)
        self.assertNotEqual(self.genome1.compute_gradient(self.genome1.get_circuit(3)[0],9), 0)

        return
        if __name__ != "__main__":
            return
        
        # ignore all future warnings
        simplefilter(action='ignore', category=FutureWarning)

        global_layer_number = h.GlobalInnovationNumber()
        global_layer_number._layer_number = 3

        n_different = 10
        n_reps = 10
        n_amounts = 100
        gradients = pd.DataFrame()
        print()
        for k in range(n_different):
            layers = {0: l.Layer(0), 1:l.Layer(1), 2:l.Layer(2)}

            layers[0].add_gate(g.GateGene(0, g.GateType.ROT, 0))
            layers[0].add_gate(g.GateGene(1, g.GateType.ROT, 1))
            layers[0].add_gate(g.GateGene(2, g.GateType.CNOT, 0))
            layers[0].add_gate(g.GateGene(3, g.GateType.CNOT, 1))
            
            genome = gen.Genome.from_layers(global_layer_number, layers)
            inner_gradients = pd.DataFrame()
            for j in range(0, n_reps):
                temp = []
                for i in range(1, n_amounts+1):
                    print(k, j, i, end="    \r")
                    temp.append(genome.compute_gradient(*genome.get_circuit(2), shots = 1024*i))
                # print(temp)
                inner_gradients = pd.concat((inner_gradients,pd.DataFrame({k:temp})), axis =0)
                # print(gradients.head())
            gradients = pd.concat((gradients, inner_gradients), axis=1)
            # print(gradients[:])
        sns.lineplot(data=gradients)
        # plt.show()
        #     plt.plot(np.arange(1,n_amounts+1)*1024, gradients)
        plt.xlabel("Amount of shots")
        plt.ylabel("Estimated energy")
        plt.savefig("../figures/energy_v_shots_sns.png")
        plt.show()

    def test_crossover(self):
        self.logger.info("TestGenome.test_crossover")
        
        n_qubits = 3
        backend = 'ibm_perth'
        self.layers1[0].add_gate(self.gate1)
        self.layers1[0].add_gate(self.gate3)
        self.layers2[0].add_gate(self.gate2)
        self.layers2[0].add_gate(self.gate4)
        child = gen.Genome.crossover(self.genome1, self.genome2, n_qubits, backend)
        self.logger.debug(f"Parent 1: \n{self.genome1.get_circuit(n_qubits)[0].draw(fold=-1)}")
        self.logger.debug(f"Parent 2: \n{self.genome2.get_circuit(n_qubits)[0].draw(fold=-1)}")
        self.logger.debug(f"Child: \n{child.get_circuit(n_qubits)[0].draw(fold=-1)}")
        
        self.layers2[0].add_gate(self.gate1)
        self.layers2[0].add_gate(self.gate3)
        self.layers1[0].add_gate(self.gate2)
        self.layers1[0].add_gate(self.gate4)
        child = gen.Genome.crossover(self.genome1, self.genome2, n_qubits, backend)
        self.logger.debug(f"Parent 1: \n{self.genome1.get_circuit(n_qubits)[0].draw(fold=-1)}")
        self.logger.debug(f"Parent 2: \n{self.genome2.get_circuit(n_qubits)[0].draw(fold=-1)}")
        self.logger.debug(f"Child: \n{child.get_circuit(n_qubits)[0].draw(fold=-1)}")
        
        self.layers1[1].add_gate(self.gate1)
        self.layers1[1].add_gate(self.gate3)
        self.layers2[2].add_gate(self.gate3)
        child = gen.Genome.crossover(self.genome1, self.genome2, n_qubits, backend)
        self.logger.debug(f"Parent 1: \n{self.genome1.get_circuit(n_qubits)[0].draw(fold=-1)}")
        self.logger.debug(f"Parent 2: \n{self.genome2.get_circuit(n_qubits)[0].draw(fold=-1)}")
        self.logger.debug(f"Child: \n{child.get_circuit(n_qubits)[0].draw(fold=-1)}")

if __name__ == '__main__':
    log.QNEATLogger("test", file_level=logging.DEBUG, mode="w")
    unittest.main()