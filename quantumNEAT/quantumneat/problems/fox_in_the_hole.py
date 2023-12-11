from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qulacs import QuantumState
from qiskit.circuit import Parameter

from quantumneat.quant_lib_np import dtype, sz, Id, Z
from quantumneat.problems.fox_in_a_hole_gym import FoxInAHolev2

if TYPE_CHECKING:
    from quantumneat.configuration import QuantumNEATConfig, Circuit
    from quantumneat.genome import CircuitGenome

def Zs(n_Zs, n_qubits):
    U = np.array([1], dtype = dtype)
    for _ in range(n_Zs):
        U = np.kron(U, sz)
    for _ in range(n_Zs, n_qubits):
        U = np.kron(U, Id)    
    return U

def run_episode(circuit:Circuit, config:QuantumNEATConfig):
    n_holes = 5
    len_state = 2
    max_steps = 6
    env = FoxInAHolev2(n_holes, max_steps, len_state)
    env_state = env.reset()
    return_ = 0
    for _ in range(max_steps):
        env_state, reward, done = choose_action(circuit, env, env_state, len_state, n_holes, config.n_qubits)
        return_ += reward
        if done:
            break
    # print(returns)
    return -return_+1

def get_multiple_returns(circuit, config, n_iterations = 100):
    returns = []
    for i in range(0, n_iterations):
        returns.append(run_episode(circuit, config))
    return returns

def fitness(config, self:CircuitGenome, **kwargs):
    # self.logger.debug("fith fitness used")
    return 6-np.mean(get_multiple_returns(self.get_circuit()[0], self.config, 10))

def energy(self, circuit, parameters, config, **kwargs):
    return np.mean(get_multiple_returns(circuit, config, 100))

def choose_action(circuit:Circuit, env:FoxInAHolev2, env_state, len_state, n_holes, n_qubits):
    for i, param in enumerate(env_state):
        circuit.set_parameter(i, param)

    state = QuantumState(n_qubits)
    circuit.update_quantum_state(state)
    psi = state.get_vector()
    # operator = Zs(n_holes, len_state)
    # expval = (np.conj(psi).T @ operator @ psi).real
    # print(expval)

    expvals = []
    for i in range(n_holes):
    # for i in range(len_state, 0, -1):
        operator = Z(i, n_qubits)
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

def add_encoding_layer(config:QuantumNEATConfig, circuit:Circuit):
    if config.simulator == "qiskit":
        circuit.rx(Parameter('enc_0'), 0)
        circuit.rx(Parameter('enc_1'), 1)
    elif config.simulator == "qulacs":
        circuit.add_parametric_RX_gate(0, -1)
        circuit.add_parametric_RX_gate(1, -1)

if __name__ == "__main__":
    from qulacs import ParametricQuantumCircuit
    from quantumneat.configuration import QuantumNEATConfig
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from time import time
    from scipy.optimize import minimize

    config = QuantumNEATConfig(5, 10)
    circuit_ = ParametricQuantumCircuit(config.n_qubits)
    params = [1.1, 2.1, .4, -0.1, -2.3]
    add_encoding_layer(config, circuit_)
    for i in range(config.n_qubits):
        circuit_.add_parametric_RX_gate(i, params[i])
    circuit_.add_CNOT_gate(0, 1)
    circuit_.add_CNOT_gate(1, 2)
    circuit_.add_CNOT_gate(2, 3)
    circuit_.add_CNOT_gate(3, 4)
    circuit_.add_CNOT_gate(4, 0)
    # print(circuit)

    amounts = [1, 10, 100, 1000, 10000]
    # amounts = [1, 10, 100, 1000]
    # amounts = [1, 2, 3, 4, 5]
    # amounts = [1,2]
    repeats = 100
    save = False

    return_data = pd.DataFrame()
    time_data = pd.DataFrame()
    for i in amounts:
        mean_returns = []
        runtimes = []
        for j in range(repeats):
            starttime = time()
            returns = get_multiple_returns(circuit_, config, i)
            runtime = time()-starttime
            runtimes.append(runtime)
            mean_returns.append(np.mean(returns))
        return_data[i] = mean_returns
        time_data[i] = runtimes
        print(f"Return of {i:5} iterations: mean = {np.mean(mean_returns)}, std = {np.std(mean_returns)}, mean time = {np.mean(runtimes)}")

    # import pickle
    if save:
        np.save('return_data', return_data, allow_pickle=True)
        np.save('time_data', time_data, allow_pickle=True)
    # sns.boxplot(return_data)
    # plt.title("Returns")
    # plt.show()
    # # plt.savefig("plot1.png")
    # plt.close()
    # plt.title("Returns")
    # sns.boxplot(return_data)
    # plt.xscale("log")
    # plt.savefig("plot2.png")
    # plt.close()
    # sns.boxplot(time_data)
    # plt.title("Time")
    # # plt.savefig("plot3.png")
    # # plt.close()
    # # sns.boxplot(time_data)
    # # plt.title("Time")
    # # plt.xscale("log")
    # # plt.savefig("plot4.png")
    # # plt.close()
    # # sns.boxplot(time_data)
    # # plt.title("Time")
    # plt.yscale("log")
    # plt.show()
    # plt.savefig("plot5.png")
    # plt.close()
    # sns.boxplot(time_data)
    # plt.title("Time")
    # plt.xscale("log")
    # plt.xscale("linear")
    # # plt.yscale("log")
    # plt.savefig("plot6.png")
    # plt.close()

    return_opt = pd.DataFrame()
    time_opt = pd.DataFrame()
    for i in amounts:
        mean_returns = []
        runtimes = []
        for j in range(repeats):
            starttime = time()
            def objective(params):
                for ind, param in enumerate(params):
                    circuit_.set_parameter(ind+2, param)
                return get_multiple_returns(circuit_, config, i)
            returns = minimize(objective,params, method="COBYLA", tol=1e-4, options={'maxiter':config.optimize_energy_max_iter}).fun
            runtime = time()-starttime
            runtimes.append(runtime)
            mean_returns.append(np.mean(returns))
        return_opt[i] = mean_returns
        time_opt[i] = runtimes
        print(f"Return of {i:5} iterations: mean = {np.mean(mean_returns)}, std = {np.std(mean_returns)}, mean time = {np.mean(runtimes)}")

    if save:
        np.save('return_opt', return_opt, allow_pickle=True)
        np.save('time_opt', time_opt, allow_pickle=True)
    # sns.boxplot(return_data)
    # plt.title("Returns")
    # plt.show()
    # sns.boxplot(time_data)
    # plt.title("Time")
    # plt.yscale("log")
    # plt.show()
    # sns.boxplot(return_data, color='red')
    # sns.boxplot(return_opt, color='green')
    # plt.show()
    print(return_data.head())
    data = pd.DataFrame()
    data['type'] = np.concatenate((['no_optimization' for _ in range(repeats*len(amounts))], ['optimization' for _ in range(repeats*len(amounts))])) 
    amounts_part = np.ravel([[i for _ in range(repeats)] for i in amounts])
    data['amounts'] = np.concatenate((amounts_part, amounts_part))
    data['returns'] = np.concatenate((np.ravel([return_data[i] for i in amounts]), np.ravel([return_opt[i] for i in amounts])))
    data['times'] = np.concatenate((np.ravel([time_data[i] for i in amounts]),np.ravel([time_opt[i] for i in amounts])))
    print(data.head())
    sns.boxplot(data=data, x='amounts', y='returns', hue='type')
    plt.title("Returns")
    plt.savefig("returns.png")
    plt.close()
    plt.title("Time")
    sns.boxplot(data=data, x='amounts', y='times', hue='type')
    plt.savefig("times.png")
    plt.close()