# Add depolarizing channels to a qulacs circuit
from qulacs import ParametricQuantumCircuit
from qulacs.gate import DepolarizingNoise, TwoQubitDepolarizingNoise

def add_depolarizing_noise(circuit:ParametricQuantumCircuit, depol_prob:float):
    print(circuit.get_qubit_count())
    noisy_circuit = ParametricQuantumCircuit(circuit.get_qubit_count())
    for i in range(circuit.get_gate_count()):
        gate = circuit.get_gate(i)
        noisy_circuit.add_gate(gate)
        # print(gate)
        if "Parametric" in gate.get_name():
            noisy_circuit.add_gate(DepolarizingNoise(gate.get_target_index_list()[0], depol_prob))
        elif gate.get_name() == "CNOT":
            noisy_circuit.add_gate(TwoQubitDepolarizingNoise(gate.get_control_index_list()[0], gate.get_target_index_list()[0], depol_prob))
    return noisy_circuit
