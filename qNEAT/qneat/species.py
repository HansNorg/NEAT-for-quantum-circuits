class Species:
    def __init__(self, generation, key = None):
        self.original_generation = generation
        self.key = key
        self.last_improved = generation
        self.genomes = []
        self.representative = None
        self.fitness = None
        self.adjusted_fitness = None
        self.record_fitness = None