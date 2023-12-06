from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qulacs import QuantumState

from quantumneat.quant_lib_np import dtype, sz, Id, Z
from quantumneat.problems.fox_in_a_hole_gym import FoxInAHolev2

if TYPE_CHECKING:
    from quantumneat.configuration import QuantumNEATConfig, Circuit

def Zs(n_Zs, n_qubits):
    U = np.array([1], dtype = dtype)
    for _ in range(n_Zs):
        U = np.kron(U, sz)
    for _ in range(n_Zs, n_qubits):
        U = np.kron(U, Id)    
    return U

def run_episode(self, circuit:Circuit, config:QuantumNEATConfig):
    n_holes = 5
    len_state = config.n_qubits
    max_steps = max(2*n_holes, len_state)
    env = FoxInAHolev2(n_holes, max_steps, len_state)
    env_state = env.reset()
    return_ = 0
    for _ in range(max_steps):
        env_state, reward, done = choose_action(env, env_state, len_state, n_holes)
        return_ += reward
        if done:
            break
    # print(returns)
    return return_

def get_multiple_returns(self, circuit, config, n_iterations):
    returns = []
    for i in range(0, n_iterations):
        returns.append(run_episode(None, circuit, config))
    return returns

def choose_action(env, env_state, len_state, n_holes):
    for i, param in enumerate(env_state):
        circuit.set_parameter(i, param)

    state = QuantumState(len_state)
    circuit.update_quantum_state(state)
    psi = state.get_vector()
    # operator = Zs(n_holes, len_state)
    # expval = (np.conj(psi).T @ operator @ psi).real
    # print(expval)

    expvals = []
    for i in range(n_holes):
    # for i in range(len_state, 0, -1):
        operator = Z(i, len_state)
        expval = (np.conj(psi).T @ operator @ psi).real
        expvals.append(expval)

    # print(expvals)
    action = np.argmax(expvals)
    # print(action)
    env_state, reward, done, _ = env.step(action)
    # print(env_state)
    # print(reward)
    # print(done)
    return env_state, reward, done

if __name__ == "__main__":
    from qulacs import ParametricQuantumCircuit
    from quantumneat.configuration import QuantumNEATConfig
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from time import time

    config = QuantumNEATConfig(6, 10)
    circuit = ParametricQuantumCircuit(config.n_qubits)
    circuit.add_parametric_RX_gate(0, -1)
    circuit.add_parametric_RX_gate(1, -1)
    circuit.add_parametric_RX_gate(2, -1)
    circuit.add_parametric_RX_gate(3, -1)
    circuit.add_parametric_RX_gate(4, -1)
    circuit.add_parametric_RX_gate(5, -1)
    circuit.add_parametric_RX_gate(0, 1.1)
    circuit.add_parametric_RX_gate(1, 2.1)
    circuit.add_parametric_RX_gate(2, 0.4)
    circuit.add_parametric_RX_gate(3, -0.1)
    circuit.add_parametric_RX_gate(4, -2.3)
    circuit.add_parametric_RX_gate(5, 1.3)
    circuit.add_CNOT_gate(0, 1)
    circuit.add_CNOT_gate(1, 2)
    circuit.add_CNOT_gate(2, 3)
    circuit.add_CNOT_gate(3, 4)
    circuit.add_CNOT_gate(4, 5)
    circuit.add_CNOT_gate(5, 0)
    # print(circuit)

    return_data = pd.DataFrame()
    time_data = pd.DataFrame()
    # for i in [1, 10, 100, 1000, 10000]:
    # for i in [1, 10, 100, 1000]:
    for i in [1, 2, 3, 4, 5]:
        mean_returns = []
        runtimes = []
        for j in range(100):
            starttime = time()
            returns = get_multiple_returns(None, circuit, config, i)
            runtime = time()-starttime
            runtimes.append(runtime)
            mean_returns.append(np.mean(returns))
        return_data[i] = mean_returns
        time_data[i] = runtimes
        print(f"Return of {i:5} iterations: mean = {np.mean(mean_returns)}, std = {np.std(mean_returns)}")
    sns.boxplot(return_data)
    plt.title("Returns")
    plt.show()
    plt.title("Returns")
    sns.boxplot(return_data)
    plt.xscale("log")
    plt.show()
    sns.boxplot(time_data)
    plt.title("Time")
    plt.show()
    sns.boxplot(time_data)
    plt.title("Time")
    plt.xscale("log")
    plt.show()
    sns.boxplot(time_data)
    plt.title("Time")
    plt.yscale("log")
    plt.show()
    sns.boxplot(time_data)
    plt.title("Time")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()