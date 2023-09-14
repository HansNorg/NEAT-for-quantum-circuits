import sys
sys.path.append(sys.path[0]+"\\..")
sys.path.append(sys.path[0]+"\\..\qneat")
import unittest
import qneat.helper as h

class TestHelper(unittest.TestCase):
    def test_ising_1d_instance(self):
        if __name__ == '__main__':
            print(h.ising_1d_instance(5, 0))
            print(h.ising_1d_instance(5))

if __name__ == '__main__':
    unittest.main()