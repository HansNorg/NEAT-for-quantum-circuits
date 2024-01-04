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
    
    @abstractmethod
    def energy(self, genome:Genome) -> float:
        return None
    
    # @abstractmethod
    def solution(self) -> float:
        return None