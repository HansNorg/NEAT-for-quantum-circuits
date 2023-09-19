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

    