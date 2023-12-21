import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import quimb as q
from time import time

from quantumneat.quant_lib_np import X, Z, ZZ

def ising_1d_instance(n_qubits, seed = None):
    def rand1d(qubits):
        np.random.seed(seed)
        return [np.random.choice([+1, -1]) for _ in range(qubits)]

    # transverse field terms
    h = rand1d(n_qubits)
    # links between lines
    j = rand1d(n_qubits-1)
    return h, j

def classical_ising_hamilatonian(h_vec, J_vec):
    n_qubits = len(h_vec)
    H = 0

    for iq in range(n_qubits -1):
        H += h_vec[iq]*Z(iq, n_qubits) + J_vec[iq]*ZZ(iq, n_qubits)

    H += h_vec[n_qubits-1] * Z(n_qubits-1, n_qubits)

    return H

def transverse_ising_hamilatonian(h_vec, J_vec):
    n_qubits = len(h_vec)
    H = 0

    for iq in range(n_qubits -1):
        H += h_vec[iq]*X(iq, n_qubits) + J_vec[iq]*ZZ(iq, n_qubits)

    H += h_vec[n_qubits-1] * X(n_qubits-1, n_qubits)

    return H

def bruteforce_transverse_ising_hamiltonian(h_vec, J_vec):
    def configurations(n):
        if n == 0:
            yield [-1]
            yield [1]
        else:
            for configuration in configurations(n-1):
                yield np.concatenate(([-1], configuration))
                yield np.concatenate(([1], configuration))
    
    n_qubits = len(h_vec)
    best_energy = np.inf
    best_configurations = []
    for configuration in configurations(n_qubits):
        current_energy = ... #TODO Is this possible?
        if current_energy < best_energy:
            best_energy = current_energy
            best_configurations = [configuration]
        elif current_energy == best_energy:
            best_configurations.append(configuration)
    return best_energy, best_configurations

def exact_diagonalisation(H):
    el, ev = q.eigh(H, k=1)
    return el[0]

def bruteforceLowestValue(h,j):
    def bool_to_state(integer):
    # Convert the 1/0 of a bit to +1/-1
        return 2*int(integer)-1

    r1=list([f'{i:0{len(h)}b}' for i in range(2**len(h))])
    best_energy = np.inf
    best_configurations = []

    for k in range(0,len(r1)):
        current_energy = 0
        # r2[k] is the number of shots that have this result
        # r1[k] is the result as qubits (like 0001)
        # Energy of h
        current_energy += sum([bool_to_state(r1[k][bit_value])*h[bit_value] for bit_value in range(0,len(r1[k]))])
        # Energy of j
        current_energy += sum([bool_to_state(r1[k][bit_value])*bool_to_state(r1[k][bit_value+1])*j[bit_value] for bit_value in range(0,len(j))])
        if current_energy < best_energy:
            best_energy = current_energy
            best_configurations = [r1[k]]
        elif current_energy == best_energy:
            best_configurations.append(r1[k])

    return best_energy, best_configurations

def qubit_to_spin(state):
    def bool_to_state(integer):
        # Convert the 1/0 of a bit to +1/-1
        return 2*int(integer)-1
    return [bool_to_state(state[bit_value]) for bit_value in range(0,len(state))]

def add_encoding_layer(config, circuit):
    if config.simulator == "qiskit":
        for qubit in range(config.n_qubits):
                circuit.h(qubit)
    elif config.simulator == "qulacs":
        for qubit in range(config.n_qubits):
                circuit.add_H_gate(qubit)

if __name__ == "__main__":
    observable_h, observable_j = ising_1d_instance(5, seed = 0)
    print(observable_h, observable_j)
    H = classical_ising_hamilatonian(observable_h, observable_j)
    print(np.shape(H))
    starttime = time()
    el, ev = q.eigh(H, k=1)
    timediff = time() - starttime
    print(el, ev.T, timediff)
    starttime = time()
    exact_classical_energy, exact_classical_configurations = bruteforceLowestValue(observable_h,observable_j)
    timediff = time() - starttime
    print(exact_classical_energy, exact_classical_configurations, [qubit_to_spin(state) for state in exact_classical_configurations], timediff)
    
    # exact_classical_energy, exact_classical_configurations = bruteforce_transverse_ising_hamiltonian(observable_h,observable_j)
    # print(exact_classical_energy, exact_classical_configurations, [qubit_to_spin(state) for state in exact_classical_configurations])

    # solutions = pd.DataFrame()
    # for n_qubits in range(2, 11):
    #     solution_lengths = []
    #     length_h_string = int(4*n_qubits)
    #     length_j_string = int(4*(n_qubits-1))
    #     for i in range(1000):
    #         observable_h, observable_j = ising_1d_instance(n_qubits, seed = i)
    #         # observable_h, observable_j = [1, 1, 1, 1, 1], [-1, -1, -1, -1]
    #         # observable_h, observable_j = [1, 1, 1, 1, 1], [1, 1, 1, 1]
    #         # print(observable_h, observable_j)
    #         exact_classical_energy, exact_classical_configurations = bruteforceLowestValue(observable_h,observable_j)
    #         # print(exact_classical_energy, exact_classical_configurations, [qubit_to_spin(state) for state in exact_classical_configurations])
            
    #         # print(f"h = {str(observable_h):{length_h_string}}, j = {str(observable_j):{length_j_string}}, E = {exact_classical_energy:2}, N_solutions = {len(exact_classical_configurations):1}, "+
    #         #     f"qubits = {exact_classical_configurations}, states = {[qubit_to_spin(state) for state in exact_classical_configurations]}")
    #         print(i, end="\r")
    #         solution_lengths.append(len(exact_classical_configurations))
    #     print(f"n_qubits = {n_qubits:2}, avg_solutions = {np.mean(solution_lengths):.3f}, max_solutions = {max(solution_lengths)}")
    #     # plt.hist(solution_lengths, bins=np.arange(0, max(solution_lengths)+1)+0.5, label=f"n_qubits={n_qubits}")
    #     # print(np.histogram(solution_lengths, bins=np.arange(0, max(solution_lengths)+1)+0.5))
    #     # plt.plot(np.arange(1, max(solution_lengths)+1), np.histogram(solution_lengths, bins=np.arange(0, max(solution_lengths)+1)+0.5)[0], label=f"n_qubits={n_qubits}")
    #     solutions[f"{n_qubits}"] = solution_lengths
    # sns.histplot(solutions, multiple="dodge", bins=np.arange(0, max(solution_lengths)+1))
    # plt.xticks(range(1, max(solution_lengths)+1))
    # # plt.legend()
    # plt.show()