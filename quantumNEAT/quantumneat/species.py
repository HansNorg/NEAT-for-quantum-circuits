class Species:
    def __init__(self, generation, key = None):
        self.original_generation = generation
        self.key = key
        self.last_improved = generation
        self.genomes = []
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