import sys
sys.path.append(sys.path[0]+"\\..")
sys.path.append(sys.path[0]+"\\..\qneat")
import qneat.genome as gen
import qneat.gate as g
import qneat.helper as h
import unittest

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
        if __name__ == '__main__':
            print(genome.get_circuit(2)[0])
            print(f"tries: {tries}")

if __name__ == '__main__':
    unittest.main()