"""
change half in learner phase 
"""
import numpy as np
import logging
from utils import logger
import json
import tensorflow as tf
from .MRLCO import MRLCO
import scipy.stats as stats
import re
class MTLBO():  
    def __init__(self, policy, env, sampler, sampler_processor, 
                 batch_size, inner_batch_size, population_index): 
        self.policy = policy
        self.env = env
        self.sampler = sampler
        self.sampler_processor = sampler_processor
        self.history = {}
        self.variance_coefficient = 0.1
        self.batch_size = batch_size
        self.inner_batch_size = inner_batch_size
        self.teacher_reward = 0
        self.teacher = []
        self.teacher_sample = []
        self.objective_function_list_score=[]
        self.objective_function_list_sample=[]

    def teacher_phase(self, population, iteration, max_iterations, sess, population_index):
        logging.info('teacher_phase')        
        if len(self.objective_function_list_score) != len(population):
            for idx, student in enumerate(population):
                score, sample = self.objective_function(student, sess, population_index[idx])
                self.objective_function_list_score.append(score)
                self.objective_function_list_sample.append(sample)
            self.teacher = population[0]
            self.teacher_reward = self.objective_function_list_score[0]
            self.teacher_sample = self.objective_function_list_sample [0]
        
        # max_value, max_index = max((value, index) for index, value in enumerate(self.objective_function_list_score))
        # print(len(self.objective_function_list_score), max_value, max_index)
        # self.teacher = population[0]
        # self.teacher_reward = max_value
        # self.teacher_sample = self.objective_function_list_sample [max_index]

        mean_solution_val = np.mean(self.objective_function_list_score)
        w_start = 0.9  # or another value you choose
        w_end = 0.02    # or another value you choose
        w = w_start - (w_start - w_end) * (iteration / max_iterations)
        for idx, student in enumerate(population):
            diff = self.subtract_dicts(student, self.teacher)  # Use teacher here
            rand_num = np.random.random()
            teaching_factor = np.random.randint(1, 3)  # Either 1 or 2
            if self.objective_function_list_score[idx] > mean_solution_val:
                scaled_diff = self.scale_dict(diff, teaching_factor * rand_num)
            else:
                # Adjusting the behavior based on the current iteration
                angle = (np.pi / 2) * (iteration / max_iterations)
                scaled_diff = self.add_dicts(
                    self.scale_dict(diff, w * teaching_factor * rand_num * np.sin(angle)),
                    self.scale_dict(diff, w * teaching_factor * rand_num * np.cos(angle))
                )

            new_solution = self.add_dicts(student, scaled_diff)
            objective_function_new, sample_new = self.objective_function(new_solution, sess, 0)
            if objective_function_new > self.objective_function_list_score[idx]:
                population[idx] = new_solution
                logging.info(f'teacher_phase{idx} -- {objective_function_new} -- {self.objective_function_list_score[idx]} -- {self.teacher_reward}')
                self.objective_function_list_score[idx] = objective_function_new
                self.objective_function_list_sample[idx] = sample_new
        return self.teacher_sample

    def subtract_dicts(self, dict1, dict2):
        result = {}
        for key in dict1:
            # Remove the task policy name from the key
            processed_key = re.sub(r'task_[0-9]{1,2}_policy/', '', key)
            # Now, find the matching key in dict2
            matching_key = next((k for k in dict2 if re.sub(r'task_[0-9]{1,2}_policy/', '', k) == processed_key), None)
            if matching_key:
                result[key] = dict1[key] - dict2[matching_key]
            else:
                print(f"No matching key found for {key} in dict2")
        return result

    def scale_dict(self, dict1, scalar):
        result = {}
        for key in dict1:
            # Remove the task policy name from the key
            processed_key = re.sub(r'task_[0-9]{1,2}_policy/', '', key)
            result[processed_key] = dict1[key] * scalar
        return result
    
    def add_dicts(self, dict1, dict2):
        result = {}
        dict1_key_mapping = {re.sub(r'task_[0-9]{1,2}_policy/', '', key): key for key in dict1.keys()}
        dict2_key_mapping = {re.sub(r'task_[0-9]{1,2}_policy/', '', key): key for key in dict2.keys()}        
        for processed_key in set(dict1_key_mapping.keys()) | set(dict2_key_mapping.keys()):
            dict1_value = dict1.get(dict1_key_mapping.get(processed_key), 0)  # Get value from dict1, or 0 if key not present
            dict2_value = dict2.get(dict2_key_mapping.get(processed_key), 0)  # Get value from dict2, or 0 if key not present
            result[processed_key] = dict1_value + dict2_value
        
        return result

    def learner_phase(self, population, iteration, max_iterations, sess, population_index):
        logging.info('learner_phase')
        n = len(population)
        half_n = n // 2        
        for idx in range(half_n):  
            student = population[idx]
            j = np.random.choice([x for x in range(half_n) if x != idx])
            other_learner = population[j]
            diff = self.subtract_dicts(student, other_learner)
            rand_num = np.random.random()            
            angle = (np.pi / 2) * (iteration / max_iterations)
            
            if self.objective_function_list_score[idx] < self.objective_function_list_score[j]:
                scaled_diff = self.scale_dict(diff, rand_num)
            else:
                scaled_diff = self.scale_dict(diff, rand_num * np.cos(angle))
            
            new_solution = self.add_dicts(student, scaled_diff)
            objective_function_new , sample_new= self.objective_function(new_solution, sess, 0)
            if objective_function_new > self.objective_function_list_score[idx]:
                population[idx] = new_solution
                logging.info(f'First Group learner_phase{idx} -- {objective_function_new} -- {self.objective_function_list_score[idx]} -- {self.teacher_reward}')
                self.objective_function_list_score[idx] = objective_function_new
                self.objective_function_list_sample[idx] = sample_new

        for idx in range(half_n, n):  
            student = population[idx]
            diff = self.subtract_dicts(student, self.teacher)
            rand_num = np.random.random()            
            angle = (np.pi / 2) * (iteration / max_iterations)
            scaled_diff = self.scale_dict(diff, np.cos(angle))
            new_solution = self.add_dicts(student, scaled_diff)
            objective_function_new, sample_new = self.objective_function(new_solution, sess, 0)
            if objective_function_new > self.objective_function_list_score[idx]:
                population[idx] = new_solution
                logging.info(f'Second Group learner_phase{idx} -- {objective_function_new} -- {self.objective_function_list_score[idx]} -- {self.teacher_reward}')
                self.objective_function_list_score[idx] = objective_function_new
                self.objective_function_list_sample[idx] = sample_new
    def objective_function(self, policy_params, sess, index):
        print('objective_function')
        self.policy.set_params(policy_params, sess=sess, index=index)
        print('objective_function set_params')    
        paths = self.sampler.obtain_samples(log=False, log_prefix='')
        samples_data = self.sampler_processor.process_samples(paths, log=False, log_prefix='')
        all_rewards = np.concatenate([data['rewards'] for data in samples_data], axis=0)
        mean_reward = np.mean(all_rewards)
        
        variance = np.var(all_rewards)
        combined_metric = mean_reward - self.variance_coefficient * variance
        
        return combined_metric, samples_data[index]