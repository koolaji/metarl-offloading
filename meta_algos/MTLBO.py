import numpy as np
import logging
from utils import logger


class MTLBO:
    def __init__(self, population_size, policy, env, sampler, sampler_processor):
        self.population_size = population_size
        self.policy = policy
        self.env = env
        self.sampler = sampler
        self.sampler_processor = sampler_processor

    def teacher_phase(self, population, iteration, max_iterations):
        logging.info('teacher_phase')
        
        # Identify the teacher (best solution in the population)
        teacher = min(population, key=self.objective_function)
        
        # Compute the mean solution
        mean_solution = {}
        keys = population[0].keys()
        
        for key in keys:
            mean_solution[key] = np.mean([individual[key] for individual in population], axis=0)

        # Update the solutions using sine and cosine functions
        for idx, student in enumerate(population):
            diff = self.subtract_dicts(student, teacher)  # Use teacher here
            
            rand_num = np.random.random()
            teaching_factor = np.random.randint(1, 3)  # Either 1 or 2
            
            if self.objective_function(student) > self.objective_function(teacher):
                scaled_diff = self.scale_dict(diff, teaching_factor * rand_num)
            else:
                # Adjusting the behavior based on the current iteration
                angle = (np.pi / 2) * (iteration / max_iterations)
                scaled_diff = self.add_dicts(
                    self.scale_dict(diff, teaching_factor * rand_num * np.sin(angle)),
                    self.scale_dict(diff, teaching_factor * rand_num * np.cos(angle))
                )

            
            new_solution = self.add_dicts(student, scaled_diff)
            
            if self.objective_function(new_solution) < self.objective_function(student):
                population[idx] = new_solution



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
                scaled_diff = self.scale_dict(diff, rand_num * np.sin(angle))
            else:
                scaled_diff = self.scale_dict(diff, rand_num * np.cos(angle))
            
            new_solution = self.add_dicts(student, scaled_diff)
            
            if self.objective_function(new_solution) < self.objective_function(student):
                population[idx] = new_solution
        
        # Second Group
        teacher = min(population, key=self.objective_function)
        
        for idx in range(half_n, n):  # Only considering the second half of the population
            student = population[idx]
            
            diff = self.subtract_dicts(student, teacher)
            
            rand_num = np.random.random()
            
            # Adjusting the behavior based on the current iteration
            angle = (np.pi / 2) * (iteration / max_iterations)
            
            if self.objective_function(student) > self.objective_function(teacher):
                scaled_diff = self.scale_dict(diff, rand_num * np.sin(angle))
            else:
                scaled_diff = self.scale_dict(diff, rand_num * np.cos(angle))
            
            new_solution = self.add_dicts(student, scaled_diff)
            
            if self.objective_function(new_solution) < self.objective_function(student):
                population[idx] = new_solution
       

    def objective_function(self, policy_params):
                logging.debug('objective_function')
                self.policy.set_params(policy_params)
                paths = self.sampler.obtain_samples(log=False, log_prefix='')
                samples_data = self.sampler_processor.process_samples(paths, log="all", log_prefix='')

                all_rewards = [data['rewards'] for data in samples_data]
                rewards = np.concatenate(all_rewards, axis=0)
                return -np.mean(rewards)