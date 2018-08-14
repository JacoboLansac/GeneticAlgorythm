import pandas as pd
import numpy as np


class GA:
    def __init__(self, nslots, ngroups, population_size, elite_percent):
        self.nslots = nslots
        self.ngroups = ngroups
        self.population_size = population_size
        self.elite_percent = elite_percent  # mist be between 0-1
        self.elite_size = int(self.population_size * self.elite_percent)
        self.sample_solution = self.random_solution()

    def constraint_parameters(self, hours_per_group, max_overlap):
        self.hours_per_group = hours_per_group
        self.max_overlap = max_overlap

    def initial_population(self):
        population = []
        for i in range(self.population_size):
            individual = self.random_solution()
            population.append(individual)
        return population

    def random_solution(self):
        # TODO: for a generic GA, this may be taken out. A Super class can be the GA, and this one could inherit the GA
        groups = []
        index = ["t" + str(t) for t in range(self.nslots)]
        for g in range(self.ngroups):
            group = pd.Series(np.random.rand(self.nslots), name="G" + str(g), index=index).round().astype(int)
            groups.append(group)
        solution = pd.concat(groups, axis=1)
        return solution

    def loss_function(self, solution):
        # TODO: the penalties have to be normalized to the number of groups / slots

        # each group must have 2 hours per
        total_hours_ = ((solution.sum(axis=0) - self.hours_per_group) ** 2).sum()
        total_hours_norm = total_hours_ / self.nslots

        # no more than two groups overlapped for the same hour
        overlap_ = ((solution.sum(axis=1) - self.max_overlap) ** 2).sum()
        overlap_norm = overlap_ / self.ngroups

        loss = total_hours_norm + overlap_norm
        return loss

    def evaluate_population(self, population):
        scores = []
        for individual in population:
            score = self.loss_function(individual)
            scores.append(score)
        return scores


    def select_elite(self, population):
        scores = pd.Series(self.evaluate_population(population), index=range(self.population_size))
        sorted = scores.sort_values()

        selected_elite = []
        elite_index = sorted.iloc[:self.elite_size].index.tolist()
        for individual in population:
            idx = population.index(individual)
            try:
                if idx in elite_index:
                    selected_elite.append(individual)
            except:
                asdfsa







ga = GA(nslots=5, ngroups=3, population_size=100, elite_percent=0.2)
ga.constraint_parameters(hours_per_group=2, max_overlap=1)
initial_population = ga.initial_population()

ga.select_elite(population=initial_population)

scores = ga.evaluate_population(initial_population)
scores[2]
scores.index(min(scores))
initial_population[19]
