from __future__ import annotations

from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from quantumneat.configuration import QuantumNEATConfig
    from quantumneat.genome import Genome

class Problem(ABC):
    def __init__(self, config:QuantumNEATConfig) -> None:
        self.config = config

    @abstractmethod
    def fitness(self, genome:Genome) -> float:
        return None
    
    def gradient(self, circuit, parameters, n_parameters) -> float:
        if n_parameters == 0:
            return 0 # Prevent division by 0
        total_gradient = 0
        for ind in range(n_parameters):
            temp = parameters[ind]
            parameters[ind] += self.config.epsilon/2
            partial_gradient = self.energy(circuit, parameters, True)
            parameters[ind] -= self.config.epsilon
            partial_gradient -= self.energy(circuit, parameters, True)
            parameters[ind] = temp # Return the parameter to original value
            total_gradient += partial_gradient**2
        return total_gradient/n_parameters
    
    @abstractmethod
    def energy(self, circuit, parameters, no_optimization = False) -> float:
        return None
    
    # @abstractmethod
    def solution(self) -> float:
        return None
    
    def add_encoding_layer(self, circuit):
        pass