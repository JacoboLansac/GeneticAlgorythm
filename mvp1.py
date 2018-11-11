import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("notebook", font_scale=1.2)
sns.set_style('darkgrid', {'font.family': 'serif', 'font.serif': 'Times New Roman'})

from side_functions import timer


class GA:
    def __init__(self, nslots, ngroups, population_size, natural_selected_perc, copied_elite,
                 mutation_rate, mutation_perc):
        self.nslots = nslots
        self.ngroups = ngroups
        self.nbits = self.nslots * self.ngroups
        self.population_size = population_size
        self.natural_selected_perc = natural_selected_perc  # mist be between 0-1
        self.natural_selected_size = int(self.population_size * self.natural_selected_perc)
        self.copied_elite_perc = copied_elite
        self.copied_elite_size = self.copied_elite_perc * self.population_size
        self.new_population = []
        self.mutation_rate = mutation_rate
        self.mutation_perc = mutation_perc
        self.sample_individual = self.random_solution()

        self.historic_metrics = []
        self.iteration = 0

    def constraint_parameters(self, hours_per_group, max_overlap):
        self.hours_per_group = hours_per_group
        self.max_overlap = max_overlap

    def initial_population(self):
        print("Generating initial population...")
        population = []
        for i in range(self.population_size):
            individual = self.random_solution()
            population.append(individual)
        self.population = population
        return None

    def random_solution(self):
        # TODO: for a generic GA, this may be taken out. A Super class can be the GA, and this one could inherit the GA
        groups = []
        index = ["t" + str(t) for t in range(self.nslots)]
        for g in range(self.ngroups):
            group = pd.Series(np.random.rand(self.nslots)-0.25, name="G" + str(g), index=index).round().astype(int)
            groups.append(group)
        solution = pd.concat(groups, axis=1)
        return solution

    def loss_function(self, solution):
        # TODO: the penalties have to be normalized to the number of groups / slots

        # each group must have 2 hours per
        total_hours_ = ((solution.sum(axis=0) - self.hours_per_group) ** 2).sum()
        total_hours_norm = total_hours_ / 1#self.nslots

        # no more than two groups overlapped for the same hour
        overlap_ = ((solution.sum(axis=1) - self.max_overlap) ** 2).sum()
        overlap_norm = overlap_ / 1#self.ngroups

        loss = total_hours_norm + overlap_norm
        return loss

    def evaluate_population(self, population):
        scores = []
        for individual in population:
            score = self.loss_function(individual)
            scores.append(score)
        return scores

    def normalize_vector(self, vector, range=None):
        unbiased = vector - vector.min()
        normalized = unbiased.divide(unbiased.max())
        if range is not None:
            normalized = min(range) + normalized * (max(range) - min(range))
        return normalized

    def plot_selection_(self, sorted, elected_index):
        elected = pd.Series(index=elected_index, data=True)
        _ = pd.concat([sorted, elected], axis=1).sort_values(by=0).notnull().astype(int).reset_index()[1]
        _.plot(marker=".", ls="")
        return None

    def natural_selection(self):

        scores = self.population_scores.copy()
        sorted = scores.sort_values()

        elite_index = sorted.head(round(self.copied_elite_size)).index

        thrsholds = self.normalize_vector(sorted)
        penalized = thrsholds + np.random.rand(self.population_size) * thrsholds * 2
        natural_selected_index = penalized.sort_values().head(self.natural_selected_size).index
        self.plot_selection_(sorted, natural_selected_index)

        for_reproduction_scores = scores.loc[natural_selected_index]
        selected_for_reproduction_idx = []
        for i in sorted.index:
            if i in elite_index:
                self.new_population.append(self.population[i])
            elif i in natural_selected_index:
                selected_for_reproduction_idx.append(i)
            else:
                pass

        self.selected_for_reproduction_idx = selected_for_reproduction_idx
        self.for_reproduction_scores = for_reproduction_scores
        return None

    def generate_couples(self, elite_scores, ncouples):
        norms = self.normalize_vector(elite_scores)
        probs_ = - norms + 1
        probs_ = self.normalize_vector(probs_, (0.5, 1))
        probs = probs_.divide(probs_.sum())
        print(probs)
        parents_index = np.random.choice(probs.index, p=probs.values, size=2 * ncouples)#, replace=False)
        couples = parents_index.reshape(-1, 2, ).tolist()
        return couples

    def heatmap_population(self):

        idxs = self.population_scores.sort_values().index
        hmp = []
        for idx in idxs:
            flt = self.population[idx].values.flatten().reshape(-1,1)
            hmp.append(flt)
        hmap = pd.DataFrame(np.concatenate(hmp, axis=1))

        plt.close("all")
        sns.heatmap(hmap)
        plt.show()
        time.sleep(1)



    def breed(self, couple):

        father = self.population[couple[0]]
        mother = self.population[couple[1]]

        breed = mother.copy()
        mask = mother.copy()
        mask.loc[:] = np.random.rand(self.nbits).reshape(breed.shape).round()
        breed[mask == 1] = father[mask == 1]
        return breed

    def breeding_probabilities(self, scores):
        #TODO: introduce  a pressure factor as (top_probability / median_probability)
        norms = self.normalize_vector(scores)
        probs_ = - norms + 1
        probs_ = self.normalize_vector(probs_, (0.5, 1))
        probs = probs_.divide(probs_.sum())
        return probs

    def breed_population(self):

        nbreeds = self.population_size - len(self.new_population)

        bredprob = self.breeding_probabilities(self.population_scores)
        # bredprob.sort_values().reset_index(drop=True).plot()

        for br in range(nbreeds):
            parents_index = np.random.choice(bredprob.index, p=bredprob.values, size=2, replace=False)
            breed = self.breed(parents_index)
            self.new_population.append(breed)


        # nbreeds = int(self.population_size - self.copied_elite_size)
        # couples = self.generate_couples(self.for_reproduction_scores, nbreeds)
        #
        # # TODO: generate N couples and two breeds per couple
        #
        # for couple in couples:
        #     breed = self.breed(couple)
        #     self.new_population.append(breed)

        self.population = self.new_population.copy()
        self.new_population = []
        return None

    def mutate_indivudal(self, individual_):
        individual = individual_.copy()
        mask = self.sample_individual
        mask.loc[:] = np.random.rand(self.nbits).reshape(self.sample_individual.shape) < self.mutation_rate
        individual[mask == 1] += 1
        individual = individual.replace(2, 0)
        return individual

    def mutate_population(self):
        index_to_mutate = np.random.choice(range(self.population_size),
                                           size=int(self.mutation_perc * self.population_size))
        for i in index_to_mutate:
            self.population[i] = self.mutate_indivudal(self.population[i])
        return None

    def calculate_population_metrics(self):

        print("Generation {}".format(self.iteration))
        scores = pd.Series(self.evaluate_population(self.population), index=range(self.population_size))
        self.population_scores = scores
        self.best_individual = self.population[scores[scores == scores.min()].index[0]]

        self.iteration += 1
        metrics = {'average': self.population_scores.mean(),
                   'minimum': self.population_scores.min(),
                   'maximum': self.population_scores.max()}
        self.historic_metrics.append(pd.DataFrame(index=[self.iteration], data=metrics))
        return None

    def plot_metrics(self):
        drive = r"Z:\share\Projects\ThousandPlots/"
        metrics = pd.concat(self.historic_metrics, axis=0)
        metrics.plot()
        plt.savefig(drive + 'asdfa.png')

        self.best_individual
        self.loss_function(self.best_individual)


    @timer
    def run_genetic_algorithm(self, iterations):
        self.initial_population()

        for i in range(iterations):
            self.calculate_population_metrics()
            self.natural_selection()
            self.breed_population()
            self.mutate_population()
            # self.heatmap_population()

            #TODO: make functions to force constraints? the most importants? how does that scale for complex problems?

        return None


#===================================================================================
# Run Genetic Algorithm
#===================================================================================
ga = GA(nslots=20,
        ngroups=3,
        population_size=100,
        natural_selected_perc=0.5,
        copied_elite=0,
        mutation_perc=0.1,
        mutation_rate=0.05)

ga.constraint_parameters(hours_per_group=2,
                         max_overlap=1)

ga.run_genetic_algorithm(iterations=20)
ga.plot_metrics()
