from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from quantumNEAT.quantumneat.configuration import QuantumNEATConfig as C

class Species:
    def __init__(self, generation, config:C, key = None):
        self.config = config
        self.original_generation = generation
        self.key = key
        self.last_improved = generation
        self.genomes:list[C.Genome] = []
        self.representative = None

    def update(self, representative, genomes):
        self.representative = representative
        self.genomes = genomes

    def empty(self):
        self.genomes = []

    def add(self, genome):
        self.genomes.append(genome)

    def update_representative(self) -> bool:
        if len(self.genomes) == 0:
            return False
        self.representative = self.genomes[0]
        return True