import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import Aer, AerSimulator

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
#Inspired by QMS-Xenakis.
class GlobalInnovationNumber(metaclass=Singleton):
    '''
    Class for keeping a global innovation number.
    
    Innovation number starts at -1, such that the first one obtained from next() is 0.
    '''
    def __init__(self):
        self._innovation_number:int = -1

    def next(self):
        '''
        Get the next innovation number.

        Increments the innovation number.
        '''
        self._innovation_number += 1
        return self._innovation_number

class GlobalLayerNumber(metaclass=Singleton):
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
    
class GlobalSpeciesNumber:
    '''
    Class for keeping a global species number.
    
    Species number starts at -1, such that the first one obtained from next() is 0.
    '''
    def __init__(self):
        self._species_number:int = -1

    def next(self):
        '''
        Get the next layer number.

        Increments the layer number.
        '''
        self._species_number += 1
        return self._species_number

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
    return Aer.get_backend('aer_simulator')
    if type(backend) == str:
        provider = IBMProvider()
        available_cloud_backends = provider.backends()
        for i in available_cloud_backends: 
            if i.name == backend:
                backend = i
        if type(backend) == str:
            provider = FakeProviderForBackendV2()
            available_cloud_backends = provider.backends()
            for i in available_cloud_backends: 
                if i.name == backend:
                    backend = i
            if type(backend) == str:
                exit("the given backend is not available, exiting the system")
    return backend

def configure_circuit_to_backend(circuit, backend):
    ibm_backend = find_backend(backend)
    # ibm_backend = backend
    circuit_basis = transpile(circuit, backend=ibm_backend)
    return circuit_basis, ibm_backend

def get_circuit_properties(circuit, ibm_backend):
    complexity = 0
    circuit_error = 0
    # IBMbackend = find_backend(backend)
    # if "fake" in ibm_backend.name:
    #     ibm_backend = AerSimulator.from_backend(ibm_backend)
    for gate in circuit.data:
        # bits = [int(qubit._index) for qubit in gate.qubits]
        # circuit_error += ibm_backend.properties().gate_error(gate.operation.name,bits)
        if "c" in gate.operation.name:
        #     cx_bits = [int(gate.qubits[0]._index), int(gate.qubits[1]._index)]
        #     circuit_error += ibm_backend.properties().gate_error(gate.operation.name,cx_bits)
            complexity += 0.02
    # If a simulator is used the manual complexity value is used, otherwise the actual 2-bit circuit error is used
    if circuit_error == 0:
        return complexity
    return circuit_error