# import qiskit_nature
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_algorithms import NumPyEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.mappers import JordanWignerMapper

def get_problem(distance):
    h2_driver = PySCFDriver(
        atom = f"H 0 0 0; H 0 0 {distance}",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    h2_problem = h2_driver.run()
    return h2_problem

if __name__ == "__main__":
    h2_problem = get_problem(0.735)
    print(h2_problem)