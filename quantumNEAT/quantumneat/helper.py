import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import Aer, AerSimulator
from qulacs import DensityMatrix, QuantumState
from qulacs import ParametricQuantumCircuit

from quantumneat.quant_lib_np import Z, ZZ

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

    def next(self) -> int:
        '''
        Get the next innovation number.

        Increments the innovation number.
        '''
        self._innovation_number += 1
        return self._innovation_number
    
    def current(self) -> int:
        """Get the current innovation number."""
        return self._innovation_number
    
    def previous(self):
        '''
        Decrements the innovation number.

        Should only be used in case of failed innovations.
        '''
        self._innovation_number -= 1
    
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

def get_circuit_properties(circuit, ibm_backend=""):
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

def get_exp_val(n_qubits,circuit:ParametricQuantumCircuit,op, phys_noise = False, err_mitig = 0):
    
    expval = 0
    if phys_noise == False:
        state = QuantumState(n_qubits)
        circuit.update_quantum_state(state)
        psi = state.get_vector()
        expval += (np.conj(psi).T @ op @ psi).real
    else:
        dm = DensityMatrix(n_qubits)
        circuit.update_quantum_state(dm)
        rho = dm.get_matrix()
        if err_mitig == 0:
            expval += np.real( np.trace(op @ rho) )
        else:
            expval += np.real( np.trace(op @ rho @ rho) / np.trace(rho @ rho))
        
    return expval

def get_energy_qulacs(angles, observable, 
                      weights,circuit:ParametricQuantumCircuit, n_qubits, 
                      energy_shift, n_shots,
                      phys_noise = False 
                      ):
    """"
    Function for Qiskit energy minimization using Qulacs
    
    Input:
    angles                [array]      : list of trial angles for ansatz
    observable            [Observable] : Qulacs observable (Hamiltonian)
    circuit               [circuit]    : ansatz circuit
    n_qubits              [int]        : number of qubits
    energy_shift          [float]      : energy shift for Qiskit Hamiltonian after freezing+removing orbitals
    n_shots               [int]        : Statistical noise, number of samples taken from QC
    phys_noise            [bool]       : Whether quantum error channels are available (DM simulation) 
    
    Output:
    expval [float] : expectation value 
    
    """
        
    parameter_count_qulacs = circuit.get_parameter_count()
    # param_qulacs = [circuit.get_parameter(ind) for ind in range(parameter_count_qulacs)]    

    
    for i, j in enumerate(np.arange(parameter_count_qulacs)):
        circuit.set_parameter(j, angles[i])
          

    expval = get_exp_val(n_qubits,circuit,observable, phys_noise)
    
    shot_noise = get_shot_noise(weights, n_shots) 

    return expval + shot_noise + energy_shift

def get_shot_noise(weights, n_shots):
    
    shot_noise = 0
    
    if n_shots > 0:

        mu,sigma =0, (n_shots)**(-0.5)
        
        shot_noise +=(np.array(weights).real).T@np.random.normal(mu,sigma,len(weights))
        
    return shot_noise

def get_Ising(h_vec, J_vec):
    n_qubits = len(h_vec)
    H = 0

    for iq in range(n_qubits -1):
        H += h_vec[iq]*Z(iq, n_qubits) + J_vec[iq]*ZZ(iq, n_qubits)

    H += h_vec[n_qubits-1] * Z(n_qubits-1, n_qubits)

    return H

def get_gradient(self, circuit, n_parameters, parameters, config):
    if n_parameters == 0:
        return 0 # Prevent division by 0
    total_gradient = 0
    
    if config.simulator == 'qulacs':
        for ind in range(n_parameters):
            temp = parameters[ind]
            parameters[ind] += config.epsilon/2
            partial_gradient = get_energy_qulacs(parameters, Z, [], circuit, config.n_qubits, 0, config.n_shots, config.phys_noise)
            parameters[ind] -= config.epsilon
            partial_gradient -= get_energy_qulacs(parameters, Z, [], circuit, config.n_qubits, 0, config.n_shots, config.phys_noise)
            parameters[ind] = temp # Return the parameter to original value
            total_gradient += partial_gradient**2
    elif config.simulator == 'qiskit':
        for ind in range(n_parameters):
            temp = parameters[ind]
            parameters[ind] += config.epsilon/2
            partial_gradient = energy_from_circuit(circuit, parameters, config.n_shots)
            parameters[ind] -= config.epsilon
            partial_gradient -= energy_from_circuit(circuit, parameters, config.n_shots)
            parameters[ind] = temp # Return the parameter to original value
            total_gradient += partial_gradient**2
    return total_gradient/n_parameters
