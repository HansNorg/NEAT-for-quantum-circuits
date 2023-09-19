import sys
sys.path.append(sys.path[0]+"\\..")
sys.path.append(sys.path[0]+"\\..\qneat")
import unittest
import qneat.qNEAT as q

class TestQNEAT(unittest.TestCase):
    def setUp(self) -> None:
        self.qneat = q.QNEAT(20, 3)

    def test_init(self):
        self.assertEqual(len(self.qneat.population), 20)
        self.qneat.species[0].genomes.pop()
        self.assertEqual(len(self.qneat.population), 20)

    def test_run(self):
        self.qneat.run(0)
        

if __name__ == '__main__':
    unittest.main()