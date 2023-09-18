import numpy as np
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_provider import IBMProvider, IBMBackend
from qiskit.providers.fake_provider import FakeProviderForBackendV2

#Inspired by QMS-Xenakis.
class GlobalInnovationNumber(object):
    '''
    Class for keeping a global innovation number.
    
    Innovation number starts at 0.
    '''
    def __init__(self):
        self._innovation_number:int = 0

    def next(self):
        '''
        Get the next innovation number.

        Increments the innovation number.
        '''
        self._innovation_number += 1
        return self._innovation_number
    
    # def __new__(cls):
    #     '''
    #     Make sure there cannot be multiple GlobalInnovationNumber instances (Singleton).
    #     '''
    #     if not hasattr(cls, 'instance'):
    #         cls.instance = super(InnovationNumber, cls).__new__(cls)
    #     return cls.instance

class GlobalLayerNumber(object):
    '''
    Class for keeping a global layer number.
    
    Layer number starts at 0.
    '''
    def __init__(self):
        self._layer_number:int = 0

    def next(self):
        '''
        Get the next layer number.

        Increments the layer number.
        '''
        self._layer_number += 1
        return self._layer_number
    
    def current(self):
        return self._layer_number

def ising_1d_instance(qubits, seed = None):
    def rand1d(qubits):
        np.random.seed(seed)
        return [np.random.choice([+1, -1]) for _ in range(qubits)]

    # transverse field terms
    h = rand1d(qubits)
    # links between lines
    j = rand1d(qubits-1)
    return h, j

def compute_expected_energy(counts,h,j):
    '''
    returns the expected energy of a circuit given the counts, the 1d-ising parameters h and j
    '''
    def bool_to_state(integer):
        # Convert the 1/0 of a bit to +1/-1
        return 2*int(integer)-1
    # Get total energy of each count
    r1=list(counts.keys())
    r2=list(counts.values())
    total_energy = 0
    for k in range(0,len(r1)):
        # r2[k] is the number of shots that have this result
        # r1[k] is the result as qubits (like 0001)
        # Energy of h
        total_energy += sum([bool_to_state(r1[k][bit_value])*h[bit_value] for bit_value in range(0,len(r1[k]))])*r2[k]
        # Energy of j
        total_energy += sum([bool_to_state(r1[k][bit_value])*bool_to_state(r1[k][bit_value+1])*j[bit_value] for bit_value in range(0,len(j))])*r2[k]
    # Divide over the total count(shots)
    expectation_value = total_energy/sum(r2)
    # print(total_energy, expectation_value)
    return expectation_value

def energy_from_circuit(circuit:QuantumCircuit, parameters, shots, backend_simulator="local_qasm_simulator"):
    bound_circuit = circuit.bind_parameters(parameters)
    measurement_circuit = add_measurement(bound_circuit)
    try:
        backend_sim = AerSimulator.from_backend(backend_simulator)
    except:
        backend_sim = AerSimulator()
    counts = backend_sim.run(transpile(measurement_circuit, backend_sim), shots=shots).result().get_counts()
    h, j = ising_1d_instance(circuit.num_qubits, 0) # TODO Move, change seed
    # print(counts)
    return compute_expected_energy(counts, h, j)

def add_measurement(circuit: QuantumCircuit) -> QuantumCircuit:
    '''Mostly copied from Richard Middelkoop'''
    n_qubits = circuit.num_qubits
    # Create a Quantum Circuit
    measurement_part = QuantumCircuit(n_qubits, n_qubits)
    measurement_part.barrier(range(n_qubits))
    # map the quantum measurement to the classical bits
    measurement_part.measure(range(n_qubits), range(n_qubits))

    # The Qiskit circuit object supports composition using
    # the compose method.
    circuit.add_register(measurement_part.cregs[0])
    measurement_circuit = circuit.compose(measurement_part)
    return measurement_circuit

def find_backend(backend = "ibm_perth"):
    if type(backend) == str:
        provider = IBMProvider()
        available_cloud_backends = provider.backends()
        # print(available_cloud_backends)
        for i in available_cloud_backends: 
            if i.name == backend:
                backend = i
        if type(backend) == str:
            provider = FakeProviderForBackendV2()
            available_cloud_backends = provider.backends()
            # print(available_cloud_backends)
            for i in available_cloud_backends: 
                if i.name == backend:
                    backend = i
            if type(backend) == str:
                exit("the given backend is not available, exiting the system")
    return backend

def configure_circuit_to_backend(circuit, backend):
    ibm_backend = find_backend(backend)
    circuit_basis = transpile(circuit, backend=ibm_backend)
    return circuit_basis, ibm_backend

def get_circuit_properties(circuit, ibm_backend:IBMBackend):
    complexity = 0
    circuit_error = 0
    # IBMbackend = find_backend(backend)
    if "fake" in ibm_backend.name:
        ibm_backend = AerSimulator.from_backend(ibm_backend)
    for gate in circuit.data:
        if "c" in gate.operation.name:
            cx_bits = [int(gate.qubits[0]._index), int(gate.qubits[1]._index)]
            circuit_error += ibm_backend.properties().gate_error(gate.operation.name,cx_bits)
            complexity += 0.02
    # If a simulator is used the manual complexity value is used, otherwise the actual 2-bit circuit error is used
    if circuit_error == 0:
        circuit_error = complexity
    return circuit_error

# Unused atm
def gate_string_to_gate(circuit:QuantumCircuit, gate_string:str, n_qubits:int, qubit_seed:str, n_parameters:int = 0):
    '''
    Takes an encoded gate string and adds the corresponding gate to the given circuit.

    Parameters:
        circuit (qiskit.QuantumCircuit): circuit the gate is added to.
        gate_string (str): Encoded gate
        qubit_seed (str): Seed used for permuting the qubits to select the qubit(s) the gate acts on.
        n_parameters (int): TODO
    '''
    if n_qubits > circuit.num_qubits:
        raise ValueError("Number of qubits given exceeds number of qubits in the circuit.")
    
    qubits = np.random.RandomState(seed=int(qubit_seed,2)).permutation(n_qubits)
    ## Single Qubit gates
    if gate_string == "00000":
        # Pauli-X
        circuit.x(qubit=qubits[0])
        return circuit, n_parameters
    if gate_string == "00001":
        # originally u1 gate
        circuit.p(Parameter(str(n_parameters)),qubit=qubits[0])
        n_parameters += 1
        return circuit, n_parameters
    if gate_string == "00010":
        # originally u2 gate
        circuit.u(np.pi/2,Parameter(str(n_parameters)),Parameter(str(n_parameters+1)),qubit=qubits[0])
        n_parameters += 2
        return circuit, n_parameters
    if gate_string == "00011":
        # originally u3 gate
        circuit.u(Parameter(str(n_parameters)),Parameter(str(n_parameters+1)),Parameter(str(n_parameters+2)),qubit=qubits[0])
        n_parameters += 3
        return circuit, n_parameters
    if gate_string == "00100":
        # Pauli-Y
        circuit.y(qubit=qubits[0])
        return circuit, n_parameters
    if gate_string == "00101":
        # Pauli-Z
        circuit.z(qubit=qubits[0])
        return circuit, n_parameters
    if gate_string == "00110":
        # Hadamard
        circuit.h(qubit=qubits[0])
        return circuit, n_parameters
    if gate_string == "00111":
        # S gate
        circuit.s(qubit=qubits[0])
        return circuit, n_parameters
    if gate_string == "01000":
        # S conjugate gate
        circuit.sdg(qubit=qubits[0])
        return circuit, n_parameters
    if gate_string == "01001":
        # T gate
        circuit.t(qubit=qubits[0])
        return circuit, n_parameters
    if gate_string == "01010":
        # T conjugate gate
        circuit.tdg(qubit=qubits[0])
        return circuit, n_parameters
    if gate_string == "01011":
        # rx gate
        circuit.rx(Parameter(str(n_parameters)),qubit=qubits[0])
        n_parameters += 1
        return circuit, n_parameters
    if gate_string == "01100":
        # ry gate
        circuit.ry(Parameter(str(n_parameters)),qubit=qubits[0])
        n_parameters += 1
        return circuit, n_parameters
    if gate_string == "01101":
        # rz gate
        circuit.rz(Parameter(str(n_parameters)),qubit=qubits[0])
        n_parameters += 1
        return circuit, n_parameters
    ## Multi Qubit gates
    if gate_string == "10000":
        # Controlled NOT gate
        circuit.cx(qubits[0],qubits[1])
        return circuit, n_parameters
    if gate_string == "10001":
        # Controlled Y gate
        circuit.cy(qubits[0],qubits[1])
        return circuit, n_parameters
    if gate_string == "10011":
        # Controlled Z gate
        circuit.cz(qubits[0],qubits[1])
        return circuit, n_parameters
    if gate_string == "10100":
        # Controlled H gate
        circuit.ch(qubits[0],qubits[1])
        return circuit, n_parameters
    if gate_string == "10101":
        # Controlled rotation Z gate
        circuit.crz(Parameter(str(n_parameters)),qubits[0],qubits[1])
        n_parameters += 1
        return circuit, n_parameters
    if gate_string == "10110":
        # Controlled phase rotation gate
        circuit.cp(Parameter(str(n_parameters)),qubits[0],qubits[1])
        n_parameters += 1
        return circuit, n_parameters
    if gate_string == "10111":
        # SWAP gate
        circuit.swap(qubits[0],qubits[1])
        return circuit, n_parameters
    if gate_string == "11000":
        # Toffoli gate
        circuit.ccx(qubits[0],qubits[1],qubits[2])
        return circuit, n_parameters
    if gate_string == "11001":
        # controlled swap gate
        circuit.cswap(qubits[0],qubits[1],qubits[2])
        return circuit, n_parameters
    else:
        # Identity/no gate
        return circuit, n_parameters