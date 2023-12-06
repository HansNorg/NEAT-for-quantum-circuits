import tensorflow as tf
import tensorflow_quantum as tfq

import cirq, sympy
import numpy as np
from functools import reduce
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit
tf.get_logger().setLevel('ERROR')

def one_qubit_rotation(qubit, symbols):
    """
    Returns Cirq gates that apply a rotation of the bloch sphere about the X,
    Y and Z axis, specified by the values in `symbols`.
    """
    return [cirq.rx(symbols[0])(qubit),
            cirq.ry(symbols[1])(qubit),
            cirq.rz(symbols[2])(qubit)]

def one_qubit_rx(qubit, symbols):
    """
    Returns Cirq gates that apply a rotation of the bloch sphere about the X,
    Y and Z axis, specified by the values in `symbols`.
    """
    return [cirq.rx(symbols[0])(qubit)]

def entangling_layer_CZ(qubits):
    """
    Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    """
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cz_ops

def entangling_layer_CNOT(qubits):
    """
    Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    """
    cNOT_ops = [cirq.CNOT(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cNOT_ops += ([cirq.CNOT(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cNOT_ops

def generate_circuit(qubits, n_layers, n_inputs, RxCnot=False):
    """Prepares a data re-uploading circuit on `qubits` with `n_layers` layers."""
    # Number of qubits
    n_qubits = len(qubits)
    
    # Sympy symbols for variational angles
    if RxCnot:
        params = sympy.symbols(f'theta(0:{1*(n_layers+1)*n_qubits})')
        params = np.asarray(params).reshape((n_layers + 1, n_qubits, 1))
    else:
        params = sympy.symbols(f'theta(0:{3*(n_layers+1)*n_qubits})')
        params = np.asarray(params).reshape((n_layers + 1, n_qubits, 3))
  
    
    # Sympy symbols for encoding angles
    inputs = sympy.symbols(f'x(0:{n_layers})'+f'_(0:{n_inputs})')
    inputs = np.asarray(inputs).reshape((n_layers, n_inputs))
    
    # Define circuit
    circuit = cirq.Circuit()
    if RxCnot:
        for l in range(n_layers):
            # Variational layer Rx Cnot
            circuit += cirq.Circuit(one_qubit_rx(q, params[l, i]) for i, q in enumerate(qubits))
            circuit += entangling_layer_CNOT(qubits)
            # Encoding layer
            circuit += cirq.Circuit(cirq.rx(inputs[l, i])(qubits[i]) for i in range(n_inputs))
        
        # Last varitional layer
        circuit += cirq.Circuit(one_qubit_rx(q, params[n_layers, i]) for i,q in enumerate(qubits))
        print("Using RxCnot configuration")
    
    else:
        for l in range(n_layers):
            # Variational layer
            circuit += cirq.Circuit(one_qubit_rotation(q, params[l, i]) for i, q in enumerate(qubits))
            circuit += entangling_layer_CZ(qubits)
            # Encoding layer
            circuit += cirq.Circuit(cirq.rx(inputs[l, i])(qubits[i]) for i in range(n_inputs))

        # Last varitional layer
        circuit += cirq.Circuit(one_qubit_rotation(q, params[n_layers, i]) for i,q in enumerate(qubits))

    
    return circuit, list(params.flat), list(inputs.flat)

class ReUploadingPQC(tf.keras.layers.Layer):
    """
    Performs the transformation (s_1, ..., s_d) -> (theta_1, ..., theta_N, lmbd[1][1]s_1, ..., lmbd[1][M]s_1,
        ......., lmbd[d][1]s_d, ..., lmbd[d][M]s_d) for d=input_dim, N=theta_dim and M=n_layers.
    An activation function from tf.keras.activations, specified by `activation` ('linear' by default) is
        then applied to all lmbd[i][j]s_i.
    All angles are finally permuted to follow the alphabetical order of their symbol names, as processed
        by the ControlledPQC.
    """

    def __init__(self, qubits, n_layers, n_inputs, observables, activation="linear", name="re-uploading_PQC", RxCnot = False):
        super(ReUploadingPQC, self).__init__(name=name)
        self.n_layers = n_layers
        self.n_qubits = len(qubits)
        self.n_inputs = n_inputs

        circuit, theta_symbols, input_symbols = generate_circuit(qubits, n_layers, n_inputs, RxCnot=RxCnot)

        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name="thetas"
        )
        
        lmbd_init = tf.ones(shape=(self.n_inputs * self.n_layers,))
        self.lmbd = tf.Variable(
            initial_value=lmbd_init, dtype="float32", trainable=True, name="lambdas"
        )
        
        # Define explicit symbol order.
        symbols = [str(symb) for symb in theta_symbols + input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])
        
        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)        

    def call(self, inputs):
        # inputs[0] = encoding data for the state.
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)
        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)

        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        
        return self.computation_layer([tiled_up_circuits, joined_vars])
    
class ObservableWeights(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(ObservableWeights, self).__init__()
        self.w = tf.Variable(
            initial_value= np.random.normal(scale= 0.01, size = output_dim), dtype="float32",
            trainable=True, name="obs-weights")

    def call(self, inputs):
        x = tf.multiply(inputs, self.w)
        return x
   

def generate_model_policy(n_qubits, n_layers, n_actions, n_inputs, beta, RxCnot = False):
    """Generates a Keras model for a data re-uploading PQC policy."""
    qubits = cirq.GridQubit.rect(1, n_qubits)
    ops = [cirq.Z(q) for q in qubits]
    # ops = [(cirq.I(q)+cirq.Z(q))/2 for q in qubits]
    # if n_actions!= n_qubits:
    #     ops.extend([cirq.Z(qubits[i])*cirq.Z(qubits[i+1]) for i in range(int(n_actions - n_qubits))])
    # ops.extend([cirq.Z(qubits[0])*cirq.Z(qubits[n_qubits-1]), cirq.Z(qubits[1])*cirq.Z(qubits[n_qubits-1])])
    observables = ops # Z for every qubit

    input_tensor = tf.keras.Input(shape=(n_inputs, ), dtype=tf.dtypes.float32, name='input')
    re_uploading_pqc = ReUploadingPQC(qubits, n_layers, n_inputs, observables, RxCnot= RxCnot)([input_tensor])
    process = tf.keras.Sequential([
        ObservableWeights(n_actions),
        tf.keras.layers.Lambda(lambda x: x * beta),
        tf.keras.layers.Softmax()
    ], name="observables-policy")
    policy = process(re_uploading_pqc)
    model = tf.keras.Model(inputs=[input_tensor], outputs=policy)
    # tf.keras.utils.plot_model(model, show_shapes=True, dpi=70)


    return model

# @tf.function
def reinforce_update(states, actions, returns, model, ws, optimizers, batch_size, eta):
    states = tf.convert_to_tensor(states)
    actions = tf.convert_to_tensor(actions)
    returns = tf.convert_to_tensor(returns)

    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        logits = model(states)
        entropy_loss = -1*tf.math.reduce_sum(tf.math.multiply(logits, tf.math.log(logits)), axis=1)
        p_actions = tf.gather_nd(logits, actions)
        log_probs = tf.math.log(p_actions)
        loss = (tf.math.reduce_sum(-log_probs * returns) - eta* entropy_loss)/ batch_size
        
    grads = tape.gradient(loss, model.trainable_variables)
    for optimizer, w in zip(optimizers, ws):
        optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])


model = generate_model_policy(n_qubits = 5, n_layers =1, n_actions= 5, n_inputs = 2, beta = 1)