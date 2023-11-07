import logging

from quantumneat.configuration import QuantumNEATConfig

class QuantumNEAT:
    logger = logging.getLogger(__name__)
    def __init__(self, config:QuantumNEATConfig):
        self.config = config

        self.logger.info("QuantumNEAT Started")

        self.generation = 0
        self.population = self.config.Population()
        self.best_fitness = self.population.get_best_genome().get_fitness()

        # For experimenting only
        self.best_fitnesses = [self.best_fitness]
        self.average_fitnesses = [self.population.average_fitness]

    def run_generation(self):
        self.logger.debug(f"Best circuit: \n{self.population.get_best_genome().get_circuit()[0].draw(fold=-1)}")
        #TODO check stopping criterion
        self.population.next_generation()
        self.best_fitness = max(self.best_fitness, self.population.get_best_genome().get_fitness())
        self.best_fitnesses.append(self.best_fitness)
        self.average_fitnesses.append(self.population.average_fitness)
        
    def run(self, max_generations:int = 10):
        self.logger.info(f"Started running for {max_generations-self.generation} generations.")

        fitness_record, population_size, number_of_species = [], [], []
        while self.generation < max_generations:
            self.logger.info(f"Generation {self.generation:8}, population size: {len(self.population):8}, \
                             number of species: {len(self.population.species):4}, best fitness: {self.best_fitness:8.3f}")
            self.run_generation()    

            fitness_record.append(self.best_fitness)
            population_size.append(len(self.population.population))
            number_of_species.append(len(self.population.species))
        self.logger.info(f"Generation {self.generation:8}, population size: {len(self.population):8}, \
                         number of species: {len(self.population.species):4}, best fitness: {self.best_fitness:8.3f}")
        self.logger.info(f"Finished running.")
        return fitness_record, population_size, number_of_species, self.average_fitnesses

def main():
    settings = QuantumNEATConfig(3, 10)
    quantum_neat = QuantumNEAT(settings)
    quantum_neat.run()

if __name__ == "__main__":
    main()