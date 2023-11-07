from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantumneat.configuration import QuantumNEATConfig

class Species:
    """
    Species containing similar genomes for independant evolution.

    Variables
    ---------
    - key (Any): Unique identifier to track different species.
    - genomes (list[Genome]): Genomes belonging to the species.
    - representative (Genome): Genome representing the species.
    - original_generation (int): Initial generation the species was created in.
    - last_improved (int): Last generation the species improved its (average/max) fitness. #TODO
    """
    logger = logging.getLogger(__name__)

    def __init__(self, generation:int, config:QuantumNEATConfig, key = None):
        """
        Initialise a new species. 
        
        Parameters
        ----------
        - generation: Generation the species was created in.
        - config: Configuration with settings.
        - key (optional): Unique identifier to track species.
        """
        self.config = config
        self.original_generation = generation
        self.key = key
        self.last_improved = generation
        self.genomes:list[QuantumNEATConfig.Genome] = []
        self.representative = None

    def update(self, representative:QuantumNEATConfig.Genome, genomes:list[QuantumNEATConfig.Genome]):
        """
        Replace the representative and genomes of the species by the given ones.

        Parameters
        ----------
        - representative: New representative for the species.
        - genomes: New list of genomes belonging to the species.
        """
        self.representative = representative
        self.genomes = genomes

    def empty(self):
        """Remove all member genomes from the species."""
        self.genomes = []

    def add(self, genome:QuantumNEATConfig.Genome):
        """
        Add the given genome to the species.
        
        Parameters
        ----------
        - genome: Genome to be added.
        """
        self.genomes.append(genome)

    def update_representative(self) -> bool:
        """
        Update the representative of the species.
        
        Returns
        -------
        - (bool): Whether the update succeeded.
            If the species contains no genomes the update will fail.
        """
        if len(self.genomes) == 0:
            return False
        self.representative = self.genomes[0]
        return True