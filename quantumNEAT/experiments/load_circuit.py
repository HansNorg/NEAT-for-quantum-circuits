import pickle

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from quantumneat.configuration import QuantumNEATConfig
from quantumneat.problems.chemistry import GroundStateEnergySavedHamiltonian
from qulacs import ParametricQuantumCircuit
from qulacs.gate import DepolarizingNoise, TwoQubitDepolarizingNoise

config = QuantumNEATConfig(2, 100, simulator="qulacs", phys_noise=True)
gse = GroundStateEnergySavedHamiltonian(config, "h2")
with open("cluster/results/thesis_gs_h2_errorless_saveh_0_linear_growth_R-CNOT_2-qubits_100-population_100-optimizer-steps_0-shots_run0_circuit.pickle", mode="rb") as f:
    circuit:ParametricQuantumCircuit = pickle.load(f)
print(type(circuit))
n_parameters = circuit.get_parameter_count()
# parameters = []
# for i in range(n_parameters):
#     parameters.append(circuit.get_parameter(i))
parameters = [circuit.get_parameter(i) for i in range(n_parameters)]
print(n_parameters)

noisy_circuit = ParametricQuantumCircuit(2)
for i in range(circuit.get_gate_count()):
    gate = circuit.get_gate(i)
    noisy_circuit.add_gate(gate)
    # print(gate)
    if "Parametric" in gate.get_name():
        noisy_circuit.add_gate(DepolarizingNoise(gate.get_target_index_list()[0], config.depolarizing_noise_prob))
    elif gate.get_name() == "CNOT":
        noisy_circuit.add_gate(TwoQubitDepolarizingNoise(gate.get_control_index_list()[0], gate.get_target_index_list()[0], config.depolarizing_noise_prob))

# for i in range(circuit.get_gate_count()):
#     print("next")
#     print(circuit.get_gate(i))
#     print(noisy_circuit.get_gate(2*i))
#     print(noisy_circuit.get_gate(2*i+1))
# parameters = np.random.rand(n_parameters)
distances, energies = gse.evaluate(circuit, parameters)
noisy_distances, noisy_energies = gse.evaluate(circuit, parameters)

gse.plot_solution(color="red")
sns.scatterplot(x=distances, y=energies, label="noiseless")
sns.scatterplot(x=noisy_distances, y=noisy_energies, label="phys_noise")
plt.show()