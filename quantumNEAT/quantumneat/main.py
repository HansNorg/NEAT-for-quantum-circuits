import logging
import copy

from qulacs import ParametricQuantumCircuit

from quantumneat.configuration import QuantumNEATConfig
from quantumneat.problems.ising import bruteforceLowestValue, ising_1d_instance

class QuantumNEAT:
    logger = logging.getLogger("quantumNEAT.quantumneat.main")
    def __init__(self, config:QuantumNEATConfig):
        self.config = config

        self.logger.info("QuantumNEAT Started")

        self.population = self.config.Population(self.config)
        best_genome = self.population.get_best_genome()
        self.best_fitness = best_genome.get_fitness()

        # For experimenting only
        self.best_fitnesses = [self.best_fitness]
        instance = ising_1d_instance(self.config.n_qubits, 0)
        optimal_energy, optimal_configuration = bruteforceLowestValue(instance[0], instance[1])
        self.logger.info(f"{optimal_energy=:.2f}, {optimal_configuration=}")
        self.optimal_energy = optimal_energy
        self.best_energies = [best_genome.get_energy()]
        self.number_of_solutions = [sum([energy == self.optimal_energy for energy in self.get_energies()])]
        self.min_energies = [min(self.get_energies())]
        self.best_genomes:list[tuple[int, self.config.Genome]] = [(self.population.generation, copy.deepcopy(best_genome))]
        self.average_fitnesses = [self.population.average_fitness]
        self.population_sizes = [len(self.population.population)]
        self.number_of_species = [len(self.population.species)]

    def run_generation(self):
        # if self.config.simulator == 'qiskit':
        #     self.logger.info(f"Best circuit: \n{self.population.get_best_genome().get_circuit()[0].draw(fold=-1)}")
        # elif self.config.simulator == 'qulacs':
        #     self.logger.info(f"Best circuit: \n{self.population.get_best_genome().get_circuit()[0].to_string()}")
        #TODO check stopping criterion
        self.population.next_generation()
        # self.best_fitness = max(self.best_fitness, self.population.get_best_genome().get_fitness())
        best_genome = self.population.get_best_genome()
        if best_genome.get_fitness() > self.best_fitness:
            self.best_fitness = best_genome.get_fitness()
            self.best_genomes.append((self.population.generation, copy.deepcopy(self.population.get_best_genome())))
        self.best_fitnesses.append(self.best_fitness)
        self.best_energies.append(best_genome.get_energy())
        self.average_fitnesses.append(self.population.average_fitness)
        self.population_sizes.append(len(self.population.population))
        self.number_of_species.append(len(self.population.species))
        self.number_of_solutions.append(sum([genome.get_energy() == self.optimal_energy for genome in self.population.population]))
        self.min_energies.append(min([genome.get_energy() for genome in self.population.population]))
        
    def run(self, max_generations:int = 10):
        self.logger.info(f"Started running for {max_generations-self.population.generation} generations.")

        while self.population.generation < max_generations:
            self.logger.info(f"Generation {self.population.generation:8}, population size: {len(self.population.population):8}, number of species: {len(self.population.species):4}, best fitness: {self.best_fitness:8.3f}")
            self.run_generation()
        self.logger.info(f"Generation {self.population.generation:8}, population size: {len(self.population.population):8}, number of species: {len(self.population.species):4}, best fitness: {self.best_fitness:8.3f}")
        best_circuit_performance = self.best_genomes[-1][1].evaluate(N=1000)
        self.logger.info(f"Best circuit performance: {best_circuit_performance}")
        self.logger.info(f"Finished running.")

    def get_energies(self):
        energies = []
        for genome in self.population.population:
            energies.append(genome.get_energy())
        return energies
    
def main():
    settings = QuantumNEATConfig(3, 10)
    quantum_neat = QuantumNEAT(settings)
    quantum_neat.run()

if __name__ == "__main__":
    main()