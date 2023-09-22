import sys
sys.path.append(sys.path[0]+"\\..")
sys.path.append(sys.path[0]+"\\..\qneat")
import unittest
import qneat.qNEAT as q

class TestQNEAT(unittest.TestCase):
    def setUp(self) -> None:
        self.population_size = 100
        self.qneat = q.QNEAT(self.population_size, 3)

    def test_init(self):
        self.assertEqual(len(self.qneat.population), self.population_size)
        self.qneat.species[0].genomes.pop()
        self.assertEqual(len(self.qneat.population), self.population_size)

    def test_run(self):
        self.qneat.run(10)
        
    def test_generate_new_population(self):
        self.qneat.generate_new_population("")

if __name__ == '__main__':
    unittest.main()