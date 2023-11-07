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
import matplotlib.pyplot as plt
import os
class MTLBO():  
    def __init__(self, policy, env, sampler, sampler_processor, 
                 batch_size, inner_batch_size, population_index): 
        self.policy = policy
        self.env = env
        self.sampler = sampler
        self.sampler_processor = sampler_processor
        self.variance_coefficient = 0.1
        self.batch_size = batch_size
        self.inner_batch_size = inner_batch_size
        self.teacher_reward = 999999
        self.teacher = []
        self.teacher_sample = []
        self.objective_function_list_score=list(range(self.batch_size-1))
        self.objective_function_list_sample=list(range(self.batch_size-1))
        self.avg_rewards=0
        self.max=batch_size-1
        self.change = False
        self.resault_stat ={}
        

    def teacher_phase(self, population, iteration, max_iterations, sess, new_start):
        self.change = False
        logging.info('teacher_phase')
        if new_start:        
            for idx, student in enumerate(population):
                avg_rewards, sample, score = self.objective_function(student, sess, idx)
                self.objective_function_list_score[idx]=score
                self.objective_function_list_sample[idx]=sample
            # if   (score < self.teacher_reward): #  and self.objective_function_list_score[idx] < self.teacher_reward): 
            #         self.change =True
            #         self.teacher_reward = score
            #         self.teacher_sample = sample
            #         self.avg_rewards = avg_rewards
            #         self.teacher = student
            #         logging.info(f'teacher_phase{idx} Teacher changed{self.teacher_reward} {self.avg_rewards}')
            self.resault_stat = self.calculate_statistics_tensors(population)
            mean_solution_sample = self.resault_stat['mean']
            self.avg_rewards = avg_rewards
            min_solution_sample = self.resault_stat['min']
            avg_rewards, sample_new, objective_function_new = self.objective_function(min_solution_sample, sess=sess, index=self.max)
            self.teacher = min_solution_sample
            self.teacher_reward = objective_function_new
            self.teacher_sample = sample_new
            logging.info(f'learner_phase loop {iteration} Teacher changed {self.teacher_reward} self.avg_rewards {self.avg_rewards}')
        mean_solution_val = np.mean(self.objective_function_list_score)
        self.resault_stat = self.calculate_statistics_tensors(population)
        mean_solution_sample = self.resault_stat['mean']
        w_start = 0.9  # or another value you choose
        w_end = 0.02    # or another value you choose
        w = w_start - (w_start - w_end) * (iteration / max_iterations)
        # _, sample_mean, _ = self.objective_function(mean_solution_sample, sess, 0)
        for idx, student in enumerate(population):
            teaching_factor = round(1 + np.random.rand())
            logging.info(f'teacher_phase subtract_dicts{iteration}')
            # teaching_factor = np.random.randint(1, 3)  # Either 1 or 2
            
            if self.objective_function_list_score[idx] < mean_solution_val:
                scaled_diff = self.scale_dict(student, w)
                teacher_diff = self.subtract_dicts(self.teacher,student)
                teacher_diff = self.scale_dict(teacher_diff, np.random.random())
                new_solution = self.add_dicts(scaled_diff, teacher_diff)
                # logging.info(f'teacher_phase first part')
            else:
                tf_mean = self.scale_dict(mean_solution_sample,teaching_factor )
                diff = self.subtract_dicts(self.teacher ,tf_mean ) 
                rand_num = 2 *(np.random.random() - 0.5)
                angle = (np.pi / 2) * (iteration / max_iterations)
                x_new_first = self.scale_dict(self.add_dicts
                               (student ,self.scale_dict(self.subtract_dicts
                                                         (mean_solution_sample,  student  ),rand_num)),np.sin(angle))
                x_new_second = self.scale_dict(diff, np.cos(angle))
                new_solution = self.add_dicts(x_new_first, x_new_second)
                # logging.info(f'teacher_phase second part')
            
            
            avg_rewards, sample_new, objective_function_new = self.objective_function(new_solution, sess, self.max)
            # print(objective_function_new,self.objective_function_list_score[idx],idx)
            if avg_rewards < self.avg_rewards : #and self.objective_function_list_score[idx] < self.teacher_reward:
                    self.change =True
                    self.avg_rewards = avg_rewards
                    min_solution_sample = self.resault_stat['min']
                    avg_rewards, sample_new, objective_function_new = self.objective_function(min_solution_sample, sess=sess, index=self.max)
                    self.teacher = min_solution_sample
                    self.teacher_reward = objective_function_new
                    self.teacher_sample = sample_new
                    logging.info(f'learner_phase loop {iteration} Teacher changed {self.teacher_reward} self.avg_rewards {self.avg_rewards}')
            if objective_function_new < self.objective_function_list_score[idx]:
                    self.change = True
                    population[idx] = new_solution
                    #avg_rewards, sample_new, objective_function_new = self.objective_function(new_solution, sess=sess, index=idx)
                    logging.info(f'teacher_phase {iteration} new = {objective_function_new} old = {self.objective_function_list_score[idx]} teacher = {self.teacher_reward} self.avg_rewards {self.avg_rewards}')
                    self.objective_function_list_score[idx] = objective_function_new
                    self.objective_function_list_sample[idx] = sample_new
                    self.policy.set_params(new_solution, sess=sess, index=idx)


        return self.teacher


    def calculate_statistics_tensors(self,population):
        mean_solution_sample = {}
        max_solution_sample = {}
        min_solution_sample = {}
        stats_solution_sample={}
        all_keys = set()
        for pop in population:
            all_keys.update(pop.keys())

        # Extract the base keys (without prefix) from the unique keys
        base_keys = set([re.sub(r'task_[0-9]{1,2}_policy/', '', key) for key in all_keys])

        for base_key in base_keys:
            # Find all matching keys in the entire population for the current base_key
            matching_keys = [k for k in all_keys if re.sub(r'task_[0-9]{1,2}_policy/', '', k) == base_key]
            
            # Extract values for all matching keys from all dictionaries in the population
            values_list = []
            for key in matching_keys:
                values_list.extend([d[key] for d in population if key in d])
            
            # Compute the mean for the current base_key
            if isinstance(values_list[0], (list, np.ndarray)):
                mean_values = np.mean(values_list, axis=0).tolist()  # Compute mean along the first axis
                max_values = np.max(values_list, axis=0).tolist()  # Compute mean along the first axis
                min_values = np.min(values_list, axis=0).tolist()  # Compute mean along the first axis
            else:
                mean_values = np.mean(values_list)
                max_values = np.max(values_list)
                min_values = np.min(values_list)
            
            mean_solution_sample[base_key] = mean_values
            max_solution_sample[base_key]  = max_values
            min_solution_sample[base_key]  = min_values
        stats_solution_sample['mean'] = mean_solution_sample
        stats_solution_sample['max']  = max_solution_sample
        stats_solution_sample['min']  = min_solution_sample

        return stats_solution_sample





    def subtract_dicts(self, dict1, dict2):
        result = {}
        if not dict2:
            logging.info("subtract_dicts: dict2 is empty.")
            return result
        for key in dict1:
            # Remove the task policy name from the key
            processed_key = re.sub(r'task_[0-9]{1,2}_policy/', '', key)
            # Now, find the matching key in dict2
            matching_key = next((k for k in dict2 if re.sub(r'task_[0-9]{1,2}_policy/', '', k) == processed_key), None)
            if matching_key:
                # Convert lists to numpy arrays and perform subtraction
                result[key] = (np.array(dict1[key]) - np.array(dict2[matching_key])).tolist()
            else:
                logging.info(f"Keys in dict2: {dict2.keys()}")
                logging.info(f"Keys in dict1: {dict1.keys()}")
                logging.info(f"Processed key for {key}: {processed_key}")
                    
        return result

    def scale_dict(self, dict1, scalar):
        result = {}
        for key in dict1:
            # Remove the task policy name from the key
            processed_key = re.sub(r'task_[0-9]{1,2}_policy/', '', key)
            result[processed_key] = (np.array(dict1[key]) * scalar).tolist()            
            # logging.info(f"type {type(scalar)}{ np.array(dict1[key]).shape} {processed_key}, {np.array(result[processed_key]).shape}")
            # logging.info(f"Type of dict1[{key}]: {type(dict1[key])}")
            # logging.info(f"Value of dict1[{key}]: {dict1[key]}")
        return result
    
    def add_dicts(self, dict1, dict2):
        result = {}
        dict1_key_mapping = {re.sub(r'task_[0-9]{1,2}_policy/', '', key): key for key in dict1.keys()}
        dict2_key_mapping = {re.sub(r'task_[0-9]{1,2}_policy/', '', key): key for key in dict2.keys()}        

        for processed_key in set(dict1_key_mapping.keys()) | set(dict2_key_mapping.keys()):
            dict1_value = dict1.get(dict1_key_mapping.get(processed_key), None)
            dict2_value = dict2.get(dict2_key_mapping.get(processed_key), None)

            # Ensure both values are numpy arrays of the same shape
            if dict1_value is None:
                dict1_value = np.zeros_like(dict2_value)
            elif dict2_value is None:
                dict2_value = np.zeros_like(dict1_value)
            else:
                dict1_value = np.array(dict1_value)
                dict2_value = np.array(dict2_value)

            result[processed_key] = (dict1_value + dict2_value).tolist()
            # logging.info(f"{np.array(result[processed_key]).shape}, {processed_key}")

        return result


    def learner_phase(self, population, iteration, max_iterations, sess):
        logging.info('learner_phase')
        sorted_indices = sorted(range(len(self.objective_function_list_score)), key=lambda i: self.objective_function_list_score[i], reverse=True)

        # Split the sorted indices into top and bottom classes
        midpoint = len(sorted_indices) // 2  # Find the midpoint
        above_average = sorted_indices[:midpoint]  # Top half indices
        below_average = sorted_indices[midpoint:]  # Bottom half indices
                # Check if there is at least one member in each group
        if not above_average or not below_average:
            print(f'above_average{above_average}')
            print(f'below_average{below_average}')
            print("Population does not have a split above and below the average.")
            os.exit(1)
        for idx in below_average:  
            student = population[idx]
            try:
                j = np.random.choice([x for x in below_average if x != idx])
            except:
                 print(f'above_average{above_average}')
                 os.exit(1)
            other_learner = population[j]
            diff = self.subtract_dicts(other_learner, student)
            rand_num = np.random.random()            
            angle = (np.pi / 2) * (iteration / max_iterations)
            if self.objective_function_list_score[idx] < self.objective_function_list_score[j]:
                scaled_diff = self.add_dicts(diff, self.scale_dict(diff, np.cos(angle)))
            else:
                max_solution_sample = self.resault_stat['max']
                min_solution_sample = self.resault_stat['min']
                rand_num = (np.random.random() -0.5) * 2
                diff = self.subtract_dicts(max_solution_sample, min_solution_sample)
                scaled_diff = self.scale_dict(diff, rand_num * np.cos(angle))
            
            new_solution = scaled_diff
            avg_rewards, sample_new, objective_function_new = self.objective_function(new_solution, sess, self.max)
            if avg_rewards < self.avg_rewards:# and self.objective_function_list_score[idx] < self.teacher_reward:
                    self.change =True
                    self.avg_rewards = avg_rewards
                    min_solution_sample = self.resault_stat['min']
                    avg_rewards, sample_new, objective_function_new = self.objective_function(min_solution_sample, sess=sess, index=self.max)
                    self.teacher = min_solution_sample
                    self.teacher_reward = objective_function_new
                    self.teacher_sample = sample_new
                    logging.info(f'learner_phase loop {iteration} Teacher changed {self.teacher_reward} self.avg_rewards {self.avg_rewards} self.avg_rewards {self.avg_rewards}')
            if objective_function_new < self.objective_function_list_score[idx]:
                    self.change = True
                    population[idx] = new_solution
                    #avg_rewards, sample_new, objective_function_new = self.objective_function(new_solution, sess=sess, index=idx)
                    logging.info(f'learner_phase loop {iteration} new = {objective_function_new} old = {self.objective_function_list_score[idx]} teacher = {self.teacher_reward} ')
                    self.objective_function_list_score[idx] = objective_function_new
                    self.objective_function_list_sample[idx] = sample_new
                    self.policy.set_params(new_solution, sess=sess, index=idx)


        for idx in above_average:  
            student = population[idx]
            diff = self.subtract_dicts(self.teacher, student)
            rand_num = np.random.random()            
            angle = (np.pi / 2) * (iteration / max_iterations)
            scaled_diff = self.scale_dict(diff, np.cos(angle))
            new_solution = self.add_dicts(student, scaled_diff)
            avg_rewards, sample_new, objective_function_new = self.objective_function(new_solution, sess, self.max)
            # print(objective_function_new,self.objective_function_list_score[idx],idx)
            if avg_rewards < self.avg_rewards: # and self.objective_function_list_score[idx] < self.teacher_reward:
                    self.change =True
                    self.avg_rewards = avg_rewards
                    min_solution_sample = self.resault_stat['min']
                    avg_rewards, sample_new, objective_function_new = self.objective_function(min_solution_sample, sess=sess, index=self.max)
                    self.teacher = min_solution_sample
                    self.teacher_reward = objective_function_new
                    self.teacher_sample = sample_new
                    logging.info(f'learner_phase loop {iteration} Teacher changed {self.teacher_reward} self.avg_rewards {self.avg_rewards}')
            if objective_function_new < self.objective_function_list_score[idx]:
                    self.change = True
                    population[idx] = new_solution
                    #avg_rewards, sample_new, objective_function_new = self.objective_function(new_solution, sess=sess, index=idx)
                    logging.info(f'learner_phase loop {iteration} new = {objective_function_new} old = {self.objective_function_list_score[idx]} teacher = {self.teacher_reward}')
                    self.objective_function_list_score[idx] = objective_function_new
                    self.objective_function_list_sample[idx] = sample_new
                    self.policy.set_params(new_solution, sess=sess, index=idx)

        return self.teacher
    def objective_function(self, policy_params, sess, index, samples=None):
        self.policy.set_params(policy_params, sess=sess, index=index)
        paths = self.sampler.obtain_samples(log=False, log_prefix='')
        samples_data = self.sampler_processor.process_samples(paths, log="all", log_prefix='')
        # logging.info("################# %s ##################", index)      
        # logging.info("1 %s",-np.mean(samples_data[1]['rewards']))        
        # for i in range(len(self.objective_function_list_score)):
        #     logging.info(f"{-np.mean(samples_data[i]['rewards'])},{i},{index},{self.objective_function_list_score[i]}")  
        ret = np.array([])
        if index == self.batch_size-1:
            for i in range(len(samples_data)):
                ret = np.concatenate((ret, np.sum(samples_data[i]['rewards'], axis=-1)), axis=-1)
        else:
            for i in range(len(samples_data)-1):
                ret = np.concatenate((ret, np.sum(samples_data[i]['rewards'], axis=-1)), axis=-1)
        avg_reward = np.mean(ret)
        
        return -avg_reward, samples_data[index], -np.mean(samples_data[index]['rewards'])
