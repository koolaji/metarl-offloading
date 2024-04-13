import numpy as np

def objective_function(x):
    return np.sum(x**2)  # Example: Objective function (Sum of squares)

class TLBO:
    def __init__(self, population_size, dimensions):
        self.population_size = population_size
        self.dimensions = dimensions
        self.population = np.random.rand(population_size, dimensions) * 10  # Random initial population
        self.best_solution = None
        self.best_fitness = float('inf')
        self.max_iter = 100
    
    def run(self, max_iter):
        self.max_iter = max_iter
        for iteration in range(max_iter):
            self.teacher_phase()
            self.learner_phase(iteration)
        return self.best_solution
    
    def teacher_phase(self):
        fitness = np.apply_along_axis(objective_function, 1, self.population)
        self.best_solution = self.population[np.argmin(fitness)]
        self.best_fitness = np.min(fitness)
        
        mean = np.mean(self.population, axis=0)
        difference = self.best_solution - mean
        for i in range(self.population_size):
            self.population[i] += np.random.rand() * difference
    
    def learner_phase(self, iteration):
        for i in range(self.population_size):
            partner = np.random.randint(self.population_size)
            if i != partner:  # Ensure different learners
                if objective_function(self.population[i]) > objective_function(self.population[partner]):
                    # Adjusting using cos for decreasing impact over iterations
                    self.population[i] += (self.population[partner] - self.population[i]) * np.cos(iteration * np.pi / self.max_iter)
                else:
                    # Adjusting using sin for more variability early on
                    self.population[i] += (self.population[i] - self.population[partner]) * np.sin(iteration * np.pi / self.max_iter)

# Usage
optimizer = TLBO(10, 5)  # 10 individuals, 5 dimensions each
best_solution = optimizer.run(100)  # Run for 100 iterations
print("Best Solution:", best_solution)
