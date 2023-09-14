import sys
sys.path.append(sys.path[0]+"\\..")
sys.path.append(sys.path[0]+"\\..\qneat")
import qneat.genome as gen
import qneat.gate as g
import qneat.layer as l
import qneat.helper as h
import unittest
from qiskit import QuantumCircuit, QuantumRegister
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# import warnings filter
from warnings import simplefilter

class TestGenome(unittest.TestCase):
    def test_genome(self):
        global_layer_number = h.GlobalLayerNumber()
        genome = gen.Genome(global_layer_number)
        gate1 = g.GateGene(0, g.GateType.ROT, 0)
        gate2 = g.GateGene(1, g.GateType.ROT, 1)
        gate3 = g.GateGene(2, g.GateType.CNOT, 0)
        gate4 = g.GateGene(3, g.GateType.CNOT, 1)
        
        self.assertTrue(genome.add_gate(gate1))
        self.assertTrue(genome.add_gate(gate2))
        self.assertTrue(genome.add_gate(gate3))
        self.assertTrue(genome.add_gate(gate4))

        if __name__ == '__main__':
            tries = 0
            while not genome.add_gate(gate1):
                tries += 1
                if tries > 100:
                    break
            while not genome.add_gate(gate2):
                tries += 1
                if tries > 100:
                    break
            while not genome.add_gate(gate3):
                tries += 1
                if tries > 100:
                    break
            while not genome.add_gate(gate4):
                tries += 1
                if tries > 100:
                    break
            print(genome.get_circuit(2)[0])
            print(f"tries: {tries}")

    def test_line_up(self):
        global_layer_number = h.GlobalLayerNumber()
        
        gate1 = g.GateGene(0, g.GateType.ROT, 0)
        gate2 = g.GateGene(1, g.GateType.ROT, 1)
        gate3 = g.GateGene(2, g.GateType.CNOT, 0)
        gate4 = g.GateGene(3, g.GateType.CNOT, 1)

        layers1 = {0: l.Layer(0), 1:l.Layer(1), 2:l.Layer(2)}
        layers2 = {0: l.Layer(0), 1:l.Layer(1), 2:l.Layer(2)}
        global_layer_number._layer_number = 3

        genome1 = gen.Genome.from_layers(global_layer_number, layers1)
        genome2 = gen.Genome.from_layers(global_layer_number, layers2)

        layers1[0].add_gate(gate1)
        self.assertEqual(gen.Genome.line_up(genome1, genome2), (0, 1, 0, 0))
        layers2[0].add_gate(gate1)
        self.assertEqual(gen.Genome.line_up(genome1, genome2), (1, 0, 0, 0))
        layers2[0].add_gate(gate2)
        self.assertEqual(gen.Genome.line_up(genome1, genome2), (1, 1, 0, 0))
        layers1[0].add_gate(gate3)
        self.assertEqual(gen.Genome.line_up(genome1, genome2), (1, 2, 0, 0))
        layers2[0].add_gate(gate3)
        self.assertEqual(gen.Genome.line_up(genome1, genome2), (2, 1, 0, 0))
        layers2[0].add_gate(gate4)
        self.assertEqual(gen.Genome.line_up(genome1, genome2), (2, 2, 0, 0))
        layers1[2].add_gate(gate2)
        self.assertEqual(gen.Genome.line_up(genome1, genome2), (2, 3, 0, 0))
        layers1[2].add_gate(gate4)
        self.assertEqual(gen.Genome.line_up(genome1, genome2), (2, 4, 0, 0))
        layers2[2].add_gate(gate2)
        self.assertEqual(gen.Genome.line_up(genome1, genome2), (3, 3, 0, 0))
        layers2[2].add_gate(gate4)
        self.assertEqual(gen.Genome.line_up(genome1, genome2), (4, 2, 0, 0))

        gate5 = g.GateGene(4, g.GateType.ROT, 0)
        gate6 = g.GateGene(5, g.GateType.CNOT, 1)
        distance = g.GateGene.get_distance(gate1, gate5)[1]/3
        layers1[1].add_gate(gate1)
        layers2[1].add_gate(gate5)
        self.assertEqual(gen.Genome.line_up(genome1, genome2), (5, 2, 0, distance))
        layers1[1].add_gate(gate4)
        self.assertEqual(gen.Genome.line_up(genome1, genome2), (5, 3, 0, distance))
        layers2[1].add_gate(gate6)
        self.assertEqual(gen.Genome.line_up(genome1, genome2), (6, 2, 0, distance))

    def test_compatibility_distance(self):
        global_layer_number = h.GlobalLayerNumber()
        
        gate1 = g.GateGene(0, g.GateType.ROT, 0)
        gate2 = g.GateGene(1, g.GateType.ROT, 1)
        gate3 = g.GateGene(2, g.GateType.CNOT, 0)
        gate4 = g.GateGene(3, g.GateType.CNOT, 1)

        layers1 = {0: l.Layer(0), 1:l.Layer(1), 2:l.Layer(2)}
        layers2 = {0: l.Layer(0), 1:l.Layer(1), 2:l.Layer(2)}
        global_layer_number._layer_number = 3

        genome1 = gen.Genome.from_layers(global_layer_number, layers1)
        genome2 = gen.Genome.from_layers(global_layer_number, layers2)

        layers1[0].add_gate(gate1)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 1, 1, 1), 1)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 3, 3, 4), 3)
        layers2[0].add_gate(gate1)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 1, 1, 1), 0)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 3, 3, 4), 0)
        layers2[0].add_gate(gate2)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 1, 1, 1), .5)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 3, 3, 4), 1.5)
        layers1[0].add_gate(gate3)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 1, 1, 1), 1)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 3, 3, 4), 3)
        layers2[0].add_gate(gate3)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 1, 1, 1), 1/3)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 3, 3, 4), 1)
        layers2[0].add_gate(gate4)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 1, 1, 1), 2/4)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 3, 3, 4), 6/4)
        layers1[2].add_gate(gate2)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 1, 1, 1), 3/4)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 3, 3, 4), 9/4)
        layers1[2].add_gate(gate4)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 1, 1, 1), 4/4)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 3, 3, 4), 12/4)
        layers2[2].add_gate(gate2)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 1, 1, 1), 3/5)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 3, 3, 4), 9/5)
        layers2[2].add_gate(gate4)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 1, 1, 1), 2/6)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 3, 3, 4), 6/6)

        gate5 = g.GateGene(4, g.GateType.ROT, 0)
        gate6 = g.GateGene(5, g.GateType.CNOT, 1)
        distance = g.GateGene.get_distance(gate1, gate5)[1]/3
        layers1[1].add_gate(gate1)
        layers2[1].add_gate(gate5)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 1, 1, 1), 2/7+distance)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 3, 3, 4), 6/7+4*distance)
        layers1[1].add_gate(gate4)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 1, 1, 1), 3/7+distance)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 3, 3, 4), 9/7+4*distance)
        layers2[1].add_gate(gate6)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 1, 1, 1), 2/8+distance)
        self.assertEqual(gen.Genome.compatibility_distance(genome1, genome2, 3, 3, 4), 6/8+4*distance)

        #TODO add/update tests to account for difference between excess and disjoint

    def test_compute_gradient(self):
        # ignore all future warnings
        simplefilter(action='ignore', category=FutureWarning)

        if __name__ != "__main__":
            return
        global_layer_number = h.GlobalInnovationNumber()
        global_layer_number._layer_number = 3

        n_different = 4
        n_reps = 3
        n_amounts = 2
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
        plt.show()
        #     plt.plot(np.arange(1,n_amounts+1)*1024, gradients)
        # plt.xlabel("Amount of shots")
        # plt.ylabel("Estimated energy")
        # plt.savefig("../figures/energy_v_shots.png")
        # plt.show()

if __name__ == '__main__':
    unittest.main()