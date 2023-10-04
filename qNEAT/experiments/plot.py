import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(sys.path[0]+"\\..")
sys.path.append(sys.path[0]+"\\..\qneat")
import logging
import qneat.logger as log

class Plotter:

    def __init__(self, savename):
        self.logger = logging.getLogger("qNEAT.plot")
        self.load_data()
        self.savename = savename

    def _load_file(self, filename):
        try:
            return np.load(f"{sys.path[0]}/../results/{filename}.npy", allow_pickle=True)
        except Exception as e:
            self.logger.error("Fitness record not found")
            self.logger.debug("Fitness record not found", exc_info=True)
            return None

    def load_data(self):
        self.fitness_record = self._load_file("fitness_record")
        self.population_sizes = self._load_file("population_sizes")
        self.number_of_species = self._load_file("number_of_species")
        self.average_fitnesses = self._load_file("average_fitnesses")

    def plot_fitness_record(self, show, save):
        if type(self.fitness_record) == type(None):
            return
        plt.plot(self.fitness_record)
        self._plot("Best fitness per generation", "Generation", "Best fitness (a.u.)", "fitness_record", show, save)

    def plot_number_of_species(self, show, save):
        if type(self.number_of_species) == type(None):
            return
        plt.plot(self.number_of_species)
        self._plot("Number of species per generation", "Generation", "Number of species", "number_of_species", show, save)
    
    def plot_population_sizes(self, show, save):
        if type(self.population_sizes) == type(None):
            return
        plt.plot(self.population_sizes)
        self._plot("Size of the population per generation", "Generation", "Number of genomes", "population_sizes", show, save)

    def plot_average_fitnesses(self, show, save):
        if type(self.average_fitnesses) == type(None):
            return
        plt.plot(self.average_fitnesses)
        self._plot("Average fitness per generation", "Generation", "Average fitness (a.u.)", "average_fitnesses", show, save)

    def plot_fitness(self, show, save):
        if type(self.fitness_record) == type(None) or type(self.average_fitnesses) == type(None):
            return
        plt.plot(self.fitness_record, label = "Best")
        plt.plot(self.average_fitnesses, label = "Average")
        self._plot("Fitness progression over generations", "Generation", "Fitness (a.u.)", "fitness", show, save, legend=True)

    def plot_all(self, show, save):
        self.plot_fitness_record(show, save)
        self.plot_number_of_species(show, save)
        self.plot_population_sizes(show, save)
    
    def _plot(self, title:str, xlabel:str, ylabel:str, savename:str, show:bool, save:bool, legend:bool = False, close:bool = True):
        '''
        Sets given settings for the plot.

        Parameters
        ----------
        title (str):
            The title the plot should have.
        xlabel (str):
            Label for the x-axis of the plot.
        ylabel (str):
            Label for the y-axis of the plot.
        savename (str):
            Name used for saving the plot. This is combined with _run(self.run).
        show (bool):
            Whether to show the plot.
        save (bool):
            Whether to save the plot to a file.
        close (bool):
            Whether to close the plot.
        '''
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if legend:
            plt.legend()
        if save:
            plt.savefig(f"{sys.path[0]}/../figures/{self.savename}_{savename}")
        if show:
            plt.show()
        if close:
            plt.close()
    
def main():
    log.QNEATLogger("plot.log")
    plotter = Plotter("1st_experiments")
    plotter.plot_all(True, True)

if __name__ == "__main__":
    main()