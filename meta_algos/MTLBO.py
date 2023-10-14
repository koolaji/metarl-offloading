import numpy as np
import logging
from utils import logger
import json

class MTLBO:
    def __init__(self, population_size, policy, env, sampler, sampler_processor):
        self.population_size = population_size
        self.policy = policy
        self.env = env
        self.sampler = sampler
        self.sampler_processor = sampler_processor
        self.history = {}
        self.variance_coefficient = 0.1

    def teacher_phase(self, population, iteration, max_iterations):
        logging.info('teacher_phase')
        
        # Identify the teacher (best solution in the population)
        teacher = min(population, key=self.objective_function)
        logging.info('teacher_phase, teacher')
        
        # Compute the mean solution
        mean_solution = {}
        keys = population[0].keys()
        
        for key in keys:
            mean_solution[key] = np.mean([individual[key] for individual in population], axis=0)

        # Calculate w based on the current iteration
        w_start = 0.9  # or another value you choose
        w_end = 0.02    # or another value you choose
        w = w_start - (w_start - w_end) * (iteration / max_iterations)

        # Update the solutions using sine and cosine functions
        for idx, student in enumerate(population):
            diff = self.subtract_dicts(student, teacher)  # Use teacher here
            
            rand_num = np.random.random()
            teaching_factor = np.random.randint(1, 3)  # Either 1 or 2
            
            if self.objective_function(student) > self.objective_function(mean_solution):
                scaled_diff = self.scale_dict(diff, teaching_factor * rand_num)
            else:
                # Adjusting the behavior based on the current iteration
                angle = (np.pi / 2) * (iteration / max_iterations)
                scaled_diff = self.add_dicts(
                    self.scale_dict(diff, w * teaching_factor * rand_num * np.sin(angle)),
                    self.scale_dict(diff, w * teaching_factor * rand_num * np.cos(angle))
                )

            new_solution = self.add_dicts(student, scaled_diff)
            
            if self.objective_function(new_solution) < self.objective_function(student):
                population[idx] = new_solution
                logging.info(f'teacher_phase{idx}')





    def subtract_dicts(self,dict1, dict2):
        result = {}
        for key in dict1:
            result[key] = dict1[key] - dict2[key]
        return result

    def scale_dict(self, dict1, scalar):
        result = {}
        for key in dict1:
            result[key] = dict1[key] * scalar
        return result
    def add_dicts(self, dict1, dict2):
        result = {}
        for key in dict1:
            result[key] = dict1[key] + dict2[key]
        return result


    def learner_phase(self, population, iteration, max_iterations):
        logging.info('learner_phase')
        
        n = len(population)
        half_n = n // 2
        
        # First Group
        for idx in range(half_n):  # Only considering the first half of the population
            student = population[idx]
            
            # Randomly select another learner
            j = np.random.choice([x for x in range(half_n) if x != idx])
            other_learner = population[j]
            
            diff = self.subtract_dicts(student, other_learner)
            
            rand_num = np.random.random()
            
            # Adjusting the behavior based on the current iteration
            angle = (np.pi / 2) * (iteration / max_iterations)
            
            if self.objective_function(student) > self.objective_function(other_learner):
                scaled_diff = self.scale_dict(diff, rand_num)
            else:
                scaled_diff = self.scale_dict(diff, rand_num * np.cos(angle))
            
            new_solution = self.add_dicts(student, scaled_diff)
            
            if self.objective_function(new_solution) < self.objective_function(student):
                population[idx] = new_solution
                logging.info(f'learner_phase{idx}')

        
        # Second Group
        teacher = min(population, key=self.objective_function)
        
        for idx in range(half_n, n):  # Only considering the second half of the population
            student = population[idx]
            
            diff = self.subtract_dicts(student, teacher)
            
            rand_num = np.random.random()
            
            # Adjusting the behavior based on the current iteration
            angle = (np.pi / 2) * (iteration / max_iterations)

            scaled_diff = self.scale_dict(diff, np.cos(angle))
            
            new_solution = self.add_dicts(student, scaled_diff)
            
            if self.objective_function(new_solution) < self.objective_function(student):
                population[idx] = new_solution
                logging.info(f'learner_phase{idx}')

       

    def objective_function(self, policy_params):
        # Convert policy_params to a string
        key = json.dumps(policy_params, sort_keys=True, default=str)

        # Check if the result is already in the cache
        if key in self.history:
            return self.history[key]

        # If not, compute the result
        logging.debug('objective_function')
        self.policy.set_params(policy_params)
        paths = self.sampler.obtain_samples(log=False, log_prefix='')
        samples_data = self.sampler_processor.process_samples(paths, log=False, log_prefix='')

        all_rewards = [data['rewards'] for data in samples_data]
        rewards = np.concatenate(all_rewards, axis=0)
        result = np.mean(rewards)

        # For MTLBO, we might want to consider other metrics as well, such as variance or other moments of the reward distribution.
        # This can help the Bayesian optimization process to not only find high-reward areas but also stable ones.
        variance = np.var(rewards)
        mtlbo_result = result - self.variance_coefficient * variance  # where self.variance_coefficient is a hyperparameter

        # Store the result in the cache
        self.history[key] = mtlbo_result

        return mtlbo_result
