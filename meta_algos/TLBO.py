import numpy as np

class TLBO:
    def __init__(self, population_size, num_generations, policy, env, sampler, sample_processor):
        self.population_size = population_size
        self.num_generations = num_generations
        self.policy = policy
        self.env = env
        self.sampler = sampler
        self.sample_processor = sample_processor

    def objective_function(self, policy_params):
        # Set policy parameters
        self.policy.set_params(policy_params)
        
        # Sample trajectories using the policy
        paths = self.sampler.obtain_samples()
        
        # Process samples to get rewards
        samples_data = self.sample_processor.process_samples(paths)
        
        # Return the negative average reward (since we want to maximize reward)
        return -np.mean(samples_data['rewards'])

    def teacher_phase(self, population):
        # Compute the mean solution
        mean_solution = {}
        keys = population[0].keys()
        for key in keys:
            mean_solution[key] = np.mean([individual[key] for individual in population], axis=0)

    def subtract_dicts(self,dict1, dict2):
        result = {}
        for key in dict1:
            result[key] = dict1[key] - dict2[key]
        return result

    def scale_dict(self,d, scalar):
            """Multiply each value in the dictionary by a scalar."""
            return {key: d[key] * scalar for key in d}
    def add_dicts(self,dict1, dict2):
            """Add values in two dictionaries with the same keys."""
            return {key: dict1[key] + dict2[key] for key in dict1}

    def learner_phase(self, population):
        for i in range(self.population_size):
            # Randomly select another solution
            j = np.random.choice([x for x in range(self.population_size) if x != i])
            
            # Calculate the difference between the two solutions
            diff = self.subtract_dicts(population[i], population[j])            
            # Update the solution based on the difference
            scaled_diff = self.scale_dict(diff, np.random.random())
            new_solution = self.add_dicts(population[i], scaled_diff)
            
            # If the new solution is better, replace the old one
            if self.objective_function(new_solution) < self.objective_function(population[i]):
                population[i] = new_solution

    def optimize(self):
        # Initialize a random population of policy parameters
        population = [self.policy.get_random_params() for _ in range(self.population_size)]
        
        for generation in range(self.num_generations):
            self.teacher_phase(population)
            self.learner_phase(population)
        
        # Return the best solution found
        return min(population, key=self.objective_function)
    
    def objective_function(self, policy_params):
                self.policy.set_params(policy_params)
                paths = self.sampler.obtain_samples(log=False, log_prefix='')
                samples_data = self.sample_processor.process_samples(paths, log=False, log_prefix='')
                
                # Assuming samples_data is a list of dictionaries
                all_rewards = [data['rewards'] for data in samples_data]
                flattened_rewards = np.concatenate(all_rewards, axis=0)
                
                return -np.mean(flattened_rewards)