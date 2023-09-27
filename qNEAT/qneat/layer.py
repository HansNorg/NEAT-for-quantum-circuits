import gate as g

class Layer(object):
    #TODO Look at only adding gates in qubit order

    def __init__(self, ind:int) -> None:
        self.gates = {}
        self.ind = ind

    def add_gate(self, gate:g.GateGene) -> bool:
        if gate.gatetype.name in self.gates:
            for existing_gate in self.gates[gate.gatetype.name]:
                if gate.qubit == existing_gate.qubit:
                    # Don't add the same gate multiple times
                    return False
            self.gates[gate.gatetype.name].append(gate)
        else:
            self.gates[gate.gatetype.name] = [gate]
        return True

    def add_to_circuit(self, circuit, n_parameters):
        for gatetype in g.GateType:
            if gatetype.name in self.gates:
                for gate in self.gates[gatetype.name]:
                    circuit, n_parameters = gate.add_to_circuit(circuit, n_parameters)
        circuit.barrier()
        return circuit, n_parameters
    
    def get_gates_generator(self):
        for key in self.gates.keys():
            for gate in self.gates[key]:
                yield gate