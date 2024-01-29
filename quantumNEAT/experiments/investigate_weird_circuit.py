import numpy as np
from scipy.optimize import minimize
from qulacs import ParametricQuantumCircuit
import matplotlib.pyplot as plt

from quantumneat.problems.ising import TransverseIsing
from quantumneat.implementations.linear_growth import LinearGrowthConfig

def main():
    config = LinearGrowthConfig(5, 100)
    config.optimize_energy = True
    problem = TransverseIsing(config)
    solution = problem.solution()

    # for max_iter in range(0, 1000, 100):
    #     run(config, problem, solution, max_iter)

    for max_iter in range(0, 1000, 100):
        run(config, problem, solution, 100)
    
    plt.hlines(solution, 0, 20, label="solution")
    plt.legend()
    plt.xlabel("Number of CNOT's at the end")
    # plt.ylabel("Energy")
    plt.ylabel("Gradient")
    plt.show()

def run(config:LinearGrowthConfig, problem:TransverseIsing, solution, max_iter):
    config.optimize_energy_max_iter = max_iter
    circuit = ParametricQuantumCircuit(config.n_qubits)

    # parameters = [238.45010446, 238.21185601, 227.92147219, 248.54459276, 256.79255296, 
    #               253.21668531, 256.79229424, 249.75159017, 260.42166286, 252.93725072, 
    #               265.63970772, 259.7817515, 251.76072564, 247.50983987, 249.39489843, 
    #               250.80364307, 246.34703243, 251.39296682]
    parameters = np.random.random(size=18)*np.pi*2

    add_rot_gate(circuit, 0, parameters[0:3])
    add_rot_gate(circuit, 1, parameters[3:6])
    add_rot_gate(circuit, 2, parameters[6:9])
    add_rot_gate(circuit, 4, parameters[9:12])

    circuit.add_CNOT_gate(1, 3)

    add_rot_gate(circuit, 1, parameters[12:15])
    add_rot_gate(circuit, 3, parameters[15:18])

    circuit.add_CNOT_gate(1, 0)
    circuit.add_CNOT_gate(3, 4)

    circuit.add_CNOT_gate(1, 2)
    circuit.add_CNOT_gate(4, 3)

    circuit.add_CNOT_gate(1, 3)

    energies = []
    gradients = []
    energies.append(problem.energy(circuit, parameters))
    gradients.append(problem.gradient(circuit, parameters, 18))
    for i in range(20):
        circuit.add_CNOT_gate(1, 4)
        energies.append(problem.energy(circuit, parameters))
        gradients.append(problem.gradient(circuit, parameters, 18))
    
    plt.plot(energies+solution)#, label=f"Energy at {max_iter} COBYLA steps")
    # plt.plot(gradients, label=f"Gradient at {max_iter} COBYLA steps")

def add_rot_gate(circuit:ParametricQuantumCircuit, qubit, parameters):
    circuit.add_parametric_RX_gate(qubit, parameters[0])
    circuit.add_parametric_RY_gate(qubit, parameters[1])
    circuit.add_parametric_RZ_gate(qubit, parameters[2])


if __name__ == "__main__":
    main()