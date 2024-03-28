from time import time
import warnings
from abc import ABC, abstractmethod, abstractproperty
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Couldn't import `kahypar` - skipping from default hyper optimizer and using basic `labels` method instead.")
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import EfficientSU2, RXGate, RYGate, RZGate, CXGate
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import Target
from qiskit_algorithms import NumPyMinimumEigensolver, VQE, AdaptVQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver, GaussianForcesDriver
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, DirectMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem, HarmonicBasis
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
import quimb as q

# Verbose levels
TIME = 1
CONSTANTS = 2
ANSATZ = 3
DEBUG = 4

dtype = np.complex64

Id = np.array([[1,0],[0,1]], dtype = dtype)
sx = np.array([[0,1],[1,0]], dtype = dtype)
sy = np.array([[0,-1j],[1j,0]], dtype = dtype)
sz = np.array([[1,0],[0,-1]], dtype = dtype)

def from_string(string):
    U = np.array([1], dtype=dtype)
    for el in string:
        if el == "I":
            U = np.kron(U, Id)
        elif el == "X":
            U = np.kron(U, sx)
        elif el == "Y":
            U = np.kron(U, sy)
        elif el == "Z":
            U = np.kron(U, sz)
        else:
            raise ValueError("Pauli identifier " + el + " not found")
    return U

def hamiltonian(instance:pd.DataFrame) -> list:
    H = 0
    for string, const in instance.items():
        if string == "correction" or string == "solution":
            continue
        H += from_string(string)*const
    return H

def exact_diagonalisation(H):
    el, ev = q.eigh(H, k=1)
    return el[0]

class Hamiltonian:
    molecule:str = None

    def __init__(self, distances, verbose = TIME):
        self.verbose = verbose
        self.distances = distances
        self._solved = False
        self.data = self.get_hamiltonian_data(distances)

    def save(self, savename=""):
        self.data.to_pickle(f"{self.molecule}_hamiltonian{savename}.pkl")

    @abstractmethod
    def get_problem(self, distance) -> ElectronicStructureProblem:
        raise NotImplementedError()

    def get_mapped_problem(self, distance) -> tuple[SparsePauliOp, ElectronicStructureProblem]:
        problem = self.get_problem(distance)
        problem:ElectronicStructureProblem = FreezeCoreTransformer().transform(problem)
        # problem:ElectronicStructureProblem = FreezeCoreTransformer(remove_orbitals=[-3,-2]).transform(problem)
        return ParityMapper(problem.num_particles).map(problem.second_q_ops()[0]), problem

    def get_hamiltonian_data(self, distances):
        if self.verbose >= TIME:
            print("Getting hamiltonian data started")
            starttime = time()
        operators = []
        for distance in distances:
            operator, problem = self.get_mapped_problem(distance)
            operator_list = operator.to_list()
            operator_dict = {op[0]:op[1] for op in operator_list}
            # correction = 0
            # for const in problem.hamiltonian.constants.values():
            #     correction += const
            correction = sum(problem.hamiltonian.constants.values())
            if self.verbose >=CONSTANTS:
                print(f"{distance:.2f}, {problem.hamiltonian.constants}, {correction}")
            operator_dict["correction"] = correction #problem.hamiltonian.constants["nuclear_repulsion_energy"]
            operators.append(operator_dict)
        data = pd.DataFrame(index=distances, data=operators)
        if self.verbose >= DEBUG:
            print(data.head())
        new_index = []
        for index in data.index:
            new_index.append(np.round(index, 2))
        data.insert(0, "R", new_index)
        data.reset_index()
        data.set_index("R", inplace=True)
        if self.verbose >= DEBUG:
            print(data.head())
        if self.verbose >= TIME:
            print(f"Hamiltonian data took {time()-starttime}")
        return data

    def add_solution(self):
        solutions = self.solve_exact_diagonalisation()
        self.data["solution"] = solutions
        if self.verbose >= DEBUG:
            print(self.data.head())

    def solve_exact_diagonalisation(self, save=True, savename=""):
        if self.verbose >= TIME:
            print("Exact diagonalisation started")
            starttime = time()
        solutions = []
        for _, instance in self.data.iterrows():
            H = hamiltonian(instance)
            solutions.append(exact_diagonalisation(H)+instance["correction"])
        if save:
            np.save(f"{self.molecule}_exact_diagonalisation{savename}", solutions)
        if self.verbose >= TIME:
            print(f"Exact diagonalisation took {time()-starttime}")
        return solutions

    def solve_ground_state_eigensolver(self, save = True, savename=""):
        if self.verbose >= TIME:
            print("Ground state eigensolver started")
            starttime = time()
        solver = GroundStateEigensolver(
            JordanWignerMapper(),
            NumPyMinimumEigensolver(),
            )
        solutions = []
        for distance, instance in self.data.iterrows():
            _, problem = self.get_mapped_problem(distance)
            result = solver.solve(problem).eigenvalues
            solutions.append(result+instance["correction"])
        if save:
            np.save(f"{self.molecule}_ground_state_eigensolver{savename}", solutions)
        if self.verbose >= TIME:
            print(f"Ground state eigensolver took {time()-starttime}")
        return solutions
    
    def solve_minimum_eigensolver(self, save=True, savename=""):
        """TODO does not work yet, don't know how it works"""
        raise NotImplementedError()
        if self.verbose >= TIME:
            print("Minimum eigensolver started")
            starttime = time()
        solutions_w, solutions_wo = [], []
        for distance, _ in self.data.iterrows():
            driver = GaussianForcesDriver(logfile="hamiltonian.log")
            basis = HarmonicBasis([2, 2, 2, 2])
            vib_problem = driver.run(basis=basis)
            vib_problem.hamiltonian.truncation_order = 2

            mapper = DirectMapper()

            solver_without_filter = NumPyMinimumEigensolver()
            solver_with_filter = NumPyMinimumEigensolver(
                filter_criterion=vib_problem.get_default_filter_criterion()
            )

            gsc_wo = GroundStateEigensolver(mapper, solver_without_filter)
            result_wo = gsc_wo.solve(vib_problem)

            gsc_w = GroundStateEigensolver(mapper, solver_with_filter)
            result_w = gsc_w.solve(vib_problem)

            solutions_wo.append(result_wo)
            solutions_w.append(result_w)
        if save:
            np.save(f"{self.molecule}_minimum_eigensolver_without{savename}", solutions_wo)
            np.save(f"{self.molecule}_minimum_eigensolver_with{savename}", solutions_w)
        if self.verbose >= TIME:
            print(f"Minimum eigensolver took {time()-starttime}")
        return solutions_wo, solutions_w
    
    def solve_AdaptVQE(self, save=True, savename=""):
        if self.verbose >= TIME:
            print("AdaptVQE started")
            starttime = time()
        solutions = []
        for distance, instance in self.data.iterrows():
            problem = self.get_mapped_problem(distance)[1]
            mapper = JordanWignerMapper()
            ansatz = UCCSD(
                problem.num_spatial_orbitals,
                problem.num_particles,
                mapper,
                initial_state=HartreeFock(
                    problem.num_spatial_orbitals,
                    problem.num_particles,
                    mapper,
                ),
            )
            vqe = VQE(Estimator(), ansatz, SLSQP())
            vqe.initial_point = [0.0]*ansatz.num_parameters
            adapt_vqe = AdaptVQE(vqe)
            # adapt_vqe.supports_aux_operators = lambda: True
            solver = GroundStateEigensolver(mapper, adapt_vqe)
            result = solver.solve(problem)
            energy = result.groundenergy
            solutions.append(energy+instance["correction"])
        if save:
            np.save(f"{self.molecule}_AdaptVQE{savename}", solutions)
        if self.verbose >= TIME:
            print(f"AdaptVQE took {time()-starttime}")
        return solutions

    def solve_UCCSD(self, save=True, savename=""):
        if self.verbose >= TIME:
            print("UCCSD started")
            starttime = time()
        solutions = []
        for distance, instance in self.data.iterrows():
            problem = self.get_mapped_problem(distance)[1]
            mapper = JordanWignerMapper()
            ansatz = UCCSD(
                problem.num_spatial_orbitals,
                problem.num_particles,
                mapper,
                initial_state=HartreeFock(
                    problem.num_spatial_orbitals,
                    problem.num_particles,
                    mapper,
                ),
            )
            if self.verbose >= ANSATZ:
                print(f"{distance}, {len(ansatz.operators)}, {ansatz.operators}")
            vqe_solver = VQE(Estimator(), ansatz, SLSQP())
            vqe_solver.initial_point = [0.0]*ansatz.num_parameters
            gs_solver = GroundStateEigensolver(mapper, vqe_solver)
            result = gs_solver.solve(problem)
            energy = result.groundenergy
            solutions.append(energy+instance["correction"])
        if save:
            np.save(f"{self.molecule}_UCCSD{savename}", solutions)
        if self.verbose >= TIME:
            print(f"UCCSD took {time()-starttime}")
        return solutions

    def solve_UCCSD_new(self, save=True, savename=""):
        if self.verbose >= TIME:
            print("UCCSD new started")
            starttime = time()
        solutions = []
        for distance, instance in self.data.iterrows():
            operator, problem = self.get_mapped_problem(distance)
            # mapper = JordanWignerMapper()
            mapper = ParityMapper(problem.num_particles)
            ansatz = UCCSD(
                problem.num_spatial_orbitals,
                problem.num_particles,
                mapper,
                initial_state=HartreeFock(
                    problem.num_spatial_orbitals,
                    problem.num_particles,
                    mapper,
                ),
            )
            if self.verbose >= ANSATZ:
                print(f"{distance}, {len(ansatz.operators)}, {ansatz.operators}")
            vqe_solver = VQE(Estimator(), ansatz, SLSQP())
            vqe_solver.initial_point = [0.0]*ansatz.num_parameters
            gs_solver = GroundStateEigensolver(mapper, vqe_solver)
            result = gs_solver.solve(problem)
            energy = result.groundenergy
            solutions.append(energy+instance["correction"])
        if save:
            np.save(f"{self.molecule}_UCCSD_new{savename}", solutions)
        if self.verbose >= TIME:
            print(f"UCCSD new took {time()-starttime}")
        return solutions

    def print_UCCSD(self):
        for distance, instance in self.data.iterrows():
            operator, problem = self.get_mapped_problem(distance)
            print(f"{operator.num_qubits=}")
            print(f"{problem.num_spatial_orbitals=}")
            print(f"{problem.num_particles=}")
            N = operator.num_qubits
            target = Target(num_qubits=N)
            target.add_instruction(CXGate(), {(i,(i+1)%N):None for i in range(N)})
            target.add_instruction(RXGate(Parameter('theta')), {(i,):None for i in range(N)})
            target.add_instruction(RYGate(Parameter('theta')), {(i,):None for i in range(N)})
            target.add_instruction(RZGate(Parameter('theta')), {(i,):None for i in range(N)})
            mapper = ParityMapper(problem.num_particles)#, JordanWignerMapper()
            ansatz = UCCSD(
                problem.num_spatial_orbitals,
                problem.num_particles,
                mapper,
                # initial_state=HartreeFock(
                #     problem.num_spatial_orbitals,
                #     problem.num_particles,
                #     mapper,
                # ),
            )
            # print(ansatz.draw())
            transpiled = transpile(ansatz, target=target)
            # print(transpiled.draw())
            with open(f"{self.molecule}_UCCSD_anzats.pickle", 'wb') as f:
                pickle.dump(transpiled, f)
            # with open(f"{self.molecule}_UCCSD_anzats.pickle", 'rb') as f:
            #     print("loaded", pickle.load(f))
            return
    
    def solve_hardware_efficient(self, save = True, savename=""):
        if self.verbose >= TIME:
            print("Hardware efficient started")
            starttime = time()
        solutions = []
        for distance, instance in self.data.iterrows():
            problem = self.get_mapped_problem(distance)[0]
            mapper = JordanWignerMapper()
            ansatz = EfficientSU2(
                problem.num_qubits,
                reps=1
            )
            vqe_solver = VQE(Estimator(), ansatz, SLSQP())
            vqe_solver.initial_point = [0.0]*ansatz.num_parameters
            gs_solver = GroundStateEigensolver(mapper, vqe_solver)
            result = gs_solver.solve(problem)
            energy = result.groundenergy
            solutions.append(energy+instance["correction"])
        if save:
            np.save(f"{self.molecule}_HWE{savename}", solutions)
        if self.verbose >= TIME:
            print(f"Hardware efficient took {time()-starttime}")
        return solutions

    def solve_all(self, save = True, savename=""):
        self.exact_diagonalisation = self.solve_exact_diagonalisation(save, savename)
        self.ground_state_eigensolver = self.solve_ground_state_eigensolver(save, savename)
        self.UCCSD = self.solve_UCCSD(save, savename)
        self.adaptVQE = self.solve_AdaptVQE(save, savename)
        self._solved = True

    def plot_solvers(self, show = True, savename=""):
        if not self._solved:
            self.solve_all(False)
        distances = self.data.index
        plt.plot(distances, self.exact_diagonalisation, label="ED")
        plt.plot(distances, self.ground_state_eigensolver, label="GSE")
        plt.plot(distances, self.UCCSD, label="UCCSD")
        plt.plot(distances, self.adaptVQE, label="AdaptVQE")
        plt.legend()
        plt.xlabel("Distance (Angstrom)")
        plt.ylabel("Energy")
        plt.savefig(f"{self.molecule}_solutions{savename}.png")
        if show:
            plt.show()
        else:
            plt.close()

    def plot_solvers_difference(self, show = True, savename=""):
        if not self._solved:
            self.solve_all(False)
        distances = self.data.index
        average = [(a+b+c+d)/4 for a, b, c, d in zip(self.exact_diagonalisation, self.ground_state_eigensolver, self.UCCSD, self.adaptVQE)]
        plt.plot(distances, [a-b for a, b in zip(self.exact_diagonalisation, average)], label = "ED - avg")
        plt.plot(distances, [a-b for a, b in zip(self.ground_state_eigensolver, average)], label="GSE - avg")
        plt.plot(distances, [a-b for a, b in zip(self.UCCSD, average)], label="UCCSD - avg")
        plt.plot(distances, [a-b for a, b in zip(self.adaptVQE, average)], label="AdaptVQE - avg")
        plt.legend()
        plt.xlabel("Distance (Angstrom)")
        plt.ylabel("Energy")
        # plt.yscale("log")
        plt.savefig(f"{self.molecule}_solutions_difference{savename}.png")
        if show:
            plt.show()
        else:
            plt.close()

    def default(self, savename="", add_solution = True):
        if add_solution:
            self.add_solution()
        self.save(savename)
        self.solve_all(True, savename)
        self.plot_solvers(False, savename)
        self.plot_solvers_difference(False, savename)

class H2_Hamiltonian(Hamiltonian):
    molecule = "h2"

    def get_problem(self, distance) -> ElectronicStructureProblem:
        driver = PySCFDriver(
            atom = f"H 0 0 0; H 0 0 {distance}",
            basis = "sto3g",
            charge = 0,
            spin = 0,
            unit =DistanceUnit.ANGSTROM,
        )
        problem = driver.run()
        return problem
    
class H6_Hamiltonian(Hamiltonian):
    molecule = "h6"

    def get_problem(self, distance) -> ElectronicStructureProblem:
        driver = PySCFDriver(
            atom = f"H 0 0 0; H 0 0 {distance}; H 0 0 {2*distance}; H 0 0 {3*distance}, H 0 0 {4*distance}, H 0 0 {5*distance}",
            basis = "sto3g",
            charge = 0,
            spin = 0,
            unit =DistanceUnit.ANGSTROM,
        )
        problem = driver.run()
        return problem

class BeH2_Hamiltonian(Hamiltonian):
    molecule = "beh2"

    def add_solution(self):
        return 
    
    def solve_exact_diagonalisation(self, save=True, savename=""):
        return np.zeros(len(self.data.index))
    
    def get_problem(self, distance) -> ElectronicStructureProblem:
        driver = PySCFDriver(
            atom = f"H 0 0 0; Be 0 0 {distance}; H 0 0 {2*distance}",
            basis = "sto3g",
            charge = 0,
            spin = 0,
            unit =DistanceUnit.ANGSTROM,
        )
        problem = driver.run()
        return problem
    
class LiH_Hamiltonian(Hamiltonian):
    molecule = "lih"

    def get_problem(self, distance) -> ElectronicStructureProblem:
        driver = PySCFDriver(
            atom = f"Li 0 0 0; H 0 0 {distance}",
            basis = "sto3g",
            charge = 0,
            spin = 0,
            unit =DistanceUnit.ANGSTROM,
        )
        problem = driver.run()
        return problem

if __name__ == "__main__":
    from argparse import ArgumentParser
    argparser = ArgumentParser()
    argparser.add_argument("molecule", nargs="+")
    argparser.add_argument("--print", action="store_true")
    argparser.add_argument("--savename", type=str, default="")
    argparser.add_argument("-v", "--verbose", action="count", default=0)
    args = argparser.parse_args()

    for molecule in args.molecule:
        if molecule == "h2":
            hamil = H2_Hamiltonian
            distances = np.arange(0.2, 2.9, 0.05)
        elif molecule == "h6":
            hamil = H6_Hamiltonian
            distances = np.arange(0.2, 2.9, 0.05)
        elif molecule == "beh2":
            hamil = BeH2_Hamiltonian
            distances = np.arange(0.2, 2.51, 0.1)
        elif molecule == "lih":
            hamil = LiH_Hamiltonian
            distances = np.arange(1, 2.76, 0.25)
        else:
            print(f"Molecule {molecule} not implemented.")
            continue
        hamil = hamil(distances, verbose=args.verbose)
        if args.print:
            hamil.print_UCCSD()
            # hamil.solve_UCCSD_new()
        else:
            hamil.default(args.savename)
        # ed = hamil.solve_exact_diagonalisation()
        # he = hamil.solve_hardware_efficient()transpiled, 
        # plt.plot(ed)
        # plt.plot(he)
        # plt.show()
        # plt.plot(he-ed)
        # plt.show()