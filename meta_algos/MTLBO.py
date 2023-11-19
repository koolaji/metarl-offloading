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
        self.batch_size = batch_size
        self.inner_batch_size = inner_batch_size
        self.teacher_reward = 999999
        self.teacher = []
        self.teacher_sample = []
        self.objective_function_list_score=list(range(self.batch_size))
        self.objective_function_list_sample=list(range(self.batch_size))
        self.avg_rewards=9999999
        self.change = False
        self.resault_stat ={}
        self.w_start = 0.9  
        self.w_end = 0.01
        
    def teacher_phase(self, population, iteration, max_iterations, sess, new_start):
        self.change = False
        logging.info('teacher_phase')
        if new_start:
            self.avg_rewards=9999999        
            for idx, student in enumerate(population):
                avg_rewards, sample, score = self.objective_function(student, sess, idx)
                self.objective_function_list_score[idx]=score
                self.objective_function_list_sample[idx]=sample
                if avg_rewards < self.avg_rewards :
                    self.resault_stat = self.calculate_statistics_tensors(population)
                    sample_new, objective_function_new = self.update_teacher(iteration, sess, avg_rewards, population)

        self.resault_stat = self.calculate_statistics_tensors(population)
        mean_solution_val = np.mean(self.objective_function_list_score)
        mean_solution_sample = self.resault_stat['mean']
  
        w = self.w_start - (self.w_start - self.w_end) * (iteration / max_iterations)
        for idx, student in enumerate(population):
            teaching_factor = round(1 + np.random.rand())

            if self.objective_function_list_score[idx] < mean_solution_val:
                scaled_diff = self.scale_dict(student, w)
                teacher_diff = self.subtract_dicts(self.teacher,student)
                teacher_diff = self.scale_dict(teacher_diff, np.random.random())
                new_solution = self.add_dicts(scaled_diff, teacher_diff)
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
            
            avg_rewards, sample_new, objective_function_new = self.objective_function(new_solution, sess, idx)
            if avg_rewards < self.avg_rewards :
                    sample_new, objective_function_new = self.update_teacher(iteration, sess, avg_rewards, population)
            if objective_function_new < self.objective_function_list_score[idx]:
                    self.update_student(population, iteration, sess, idx, sample_new, objective_function_new, new_solution)

    def update_student(self, population, iteration, sess, idx, sample_new, objective_function_new, new_solution):
        self.change = True
        population[idx] = new_solution
        logging.info(f'{iteration} new = {objective_function_new} old = {self.objective_function_list_score[idx]} teacher = {self.teacher_reward} self.avg_rewards {self.avg_rewards}')
        self.objective_function_list_score[idx] = objective_function_new
        self.objective_function_list_sample[idx] = sample_new
        self.policy.set_params(new_solution, sess=sess, index=idx)

    def update_teacher(self, iteration, sess, avg_rewards,population ):
        self.change =True
        self.avg_rewards = avg_rewards
        solution_sample = self.calculate_weighted_teacher(population)
        avg_rewards, sample_new, objective_function_new = self.objective_function(solution_sample, sess=sess, index=self.batch_size-1)
        # if objective_function_new < self.teacher_reward :
        self.teacher = solution_sample
        self.teacher_reward = objective_function_new
        logging.info(f'{iteration} Teacher changed {self.teacher_reward} self.avg_rewards {self.avg_rewards}')
        return sample_new,objective_function_new

    def calculate_statistics_tensors(self,population):
        mean_solution_sample = {}
        max_solution_sample = {}
        min_solution_sample = {}
        stats_solution_sample={}
        all_keys = set()
        for pop in population:
            all_keys.update(pop.keys())
        base_keys = set([re.sub(r'task_[0-9]{1,2}_policy/', '', key) for key in all_keys])

        for base_key in base_keys:
            matching_keys = [k for k in all_keys if re.sub(r'task_[0-9]{1,2}_policy/', '', k) == base_key]            
            values_list = []
            for key in matching_keys:
                values_list.extend([d[key] for d in population if key in d])
            
            if isinstance(values_list[0], (list, np.ndarray)):
                mean_values = np.mean(values_list, axis=0).tolist()  
                max_values = np.max(values_list, axis=0).tolist()  
                min_values = np.min(values_list, axis=0).tolist()  
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
    
    def calculate_bounds_separate(self, population, factor=3):
        lower_bounds = {}
        upper_bounds = {}
        all_keys = set()
        for pop in population:
            all_keys.update(pop.keys())
        base_keys = set([re.sub(r'task_[0-9]{1,2}_policy/', '', key) for key in all_keys])

        for base_key in base_keys:
            matching_keys = [k for k in all_keys if re.sub(r'task_[0-9]{1,2}_policy/', '', k) == base_key]
            values_list = []
            for key in matching_keys:
                values_list.extend([d[key] for d in population if key in d])

            if isinstance(values_list[0], (list, np.ndarray)):
                values_array = np.array(values_list)
            else:
                values_array = np.array(values_list, dtype=float)

            stddev = np.std(values_array, axis=0)
            mean = np.mean(values_array, axis=0)

            lower_bounds[base_key] = (mean - factor * stddev).tolist()
            upper_bounds[base_key] = (mean + factor * stddev).tolist()

        return lower_bounds, upper_bounds
    
    def calculate_weighted_teacher(self, population):
        weighted_teacher = {}
        for index, policy in enumerate(population):
            for key in policy:
                if key not in weighted_teacher:
                    weighted_teacher[key] = 0 
                weight = self.get_weight_for_policy(index) 
                weighted_teacher[key] += (weight * np.array(policy[key]))
        return weighted_teacher
    

    # def get_weight_for_policy(self, policy_index):
    #     if not isinstance(policy_index, int):
    #         raise TypeError("policy_index must be an integer")
    #     performance_metric = self.objective_function_list_score[policy_index]
    #     weight = performance_metric / (sum(self.objective_function_list_score)) 
    #     return weight
    def get_weight_for_policy(self, policy_index):
        if not isinstance(policy_index, int):
            raise TypeError("policy_index must be an integer")

        # Ensure that the sum of scores is not zero to avoid division by zero
        sum_of_scores = sum(self.objective_function_list_score)
        if sum_of_scores == 0:
            sum_of_scores = 1e-6  # small epsilon value to avoid division by zero

        # Shift rewards to ensure they are positive if they can be negative
        min_reward = min(self.objective_function_list_score)
        if min_reward < 0:
            shift_value = abs(min_reward)
            adjusted_scores = [score + shift_value for score in self.objective_function_list_score]
            performance_metric = adjusted_scores[policy_index]
        else:
            performance_metric = self.objective_function_list_score[policy_index]

        # Calculate weight
        weight = performance_metric / sum_of_scores

        # Optionally, you can apply a softmax for scaling if the range of rewards is very large
        # softmax_scores = np.exp(adjusted_scores) / np.sum(np.exp(adjusted_scores), axis=0)
        # weight = softmax_scores[policy_index]

        return weight



    def subtract_dicts(self, dict1, dict2):
        result = {}
        if not dict2:
            return result
        for key in dict1:
            processed_key = re.sub(r'task_[0-9]{1,2}_policy/', '', key)
            matching_key = next((k for k in dict2 if re.sub(r'task_[0-9]{1,2}_policy/', '', k) == processed_key), None)
            if matching_key:
                result[key] = (np.array(dict1[key]) - np.array(dict2[matching_key])).tolist()
            else:
                logging.info(f"Keys in dict2: {dict2.keys()}")
                logging.info(f"Keys in dict1: {dict1.keys()}")
                logging.info(f"Processed key for {key}: {processed_key}")
        return result

    def scale_dict(self, dict1, scalar):
        result = {}
        for key in dict1:
            processed_key = re.sub(r'task_[0-9]{1,2}_policy/', '', key)
            result[processed_key] = (np.array(dict1[key]) * scalar).tolist()            
        return result
    
    def add_dicts(self, dict1, dict2):
        result = {}
        dict1_key_mapping = {re.sub(r'task_[0-9]{1,2}_policy/', '', key): key for key in dict1.keys()}
        dict2_key_mapping = {re.sub(r'task_[0-9]{1,2}_policy/', '', key): key for key in dict2.keys()}        

        for processed_key in set(dict1_key_mapping.keys()) | set(dict2_key_mapping.keys()):
            dict1_value = dict1.get(dict1_key_mapping.get(processed_key), None)
            dict2_value = dict2.get(dict2_key_mapping.get(processed_key), None)

            if dict1_value is None:
                dict1_value = np.zeros_like(dict2_value)
            elif dict2_value is None:
                dict2_value = np.zeros_like(dict1_value)
            else:
                dict1_value = np.array(dict1_value)
                dict2_value = np.array(dict2_value)

            result[processed_key] = (dict1_value + dict2_value).tolist()
        return result


    def learner_phase(self, population, iteration, max_iterations, sess):
        logging.info('learner_phase')
        sorted_indices = sorted(range(len(self.objective_function_list_score)), key=lambda i: self.objective_function_list_score[i], reverse=True)

        midpoint = len(sorted_indices) // 2  
        above_average = sorted_indices[:midpoint] 
        below_average = sorted_indices[midpoint:]  
        exploration_rate = 0.1
        for idx in below_average:  
            student = population[idx]
            if np.random.rand() < exploration_rate:
                student = {k: v + np.random.normal(0, 0.1, np.shape(v)) for k, v in student.items()}
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
                min_solution_sample , max_solution_sample= self.calculate_bounds_separate(population)
                rand_num = (np.random.random() -0.5) * 2
                diff = self.subtract_dicts(max_solution_sample, min_solution_sample)
                scaled_diff = self.scale_dict(diff, rand_num * np.cos(angle))
            
            new_solution = scaled_diff
            avg_rewards, sample_new, objective_function_new = self.objective_function(new_solution, sess, idx)
            if avg_rewards < self.avg_rewards:
                sample_new, objective_function_new = self.update_teacher(iteration, sess, avg_rewards, population)
            if objective_function_new < self.objective_function_list_score[idx]:
                self.update_student(population, iteration, sess, idx, sample_new, objective_function_new, new_solution)

        for idx in above_average:  
            student = population[idx]
            if np.random.rand() < exploration_rate:
                student = {k: v + np.random.normal(0, 0.1, np.shape(v)) for k, v in student.items()}
            diff = self.subtract_dicts(self.teacher, student)
            rand_num = np.random.random()            
            angle = (np.pi / 2) * (iteration / max_iterations)
            scaled_diff = self.scale_dict(diff, np.cos(angle))
            new_solution = self.add_dicts(student, scaled_diff)
            avg_rewards, sample_new, objective_function_new = self.objective_function(new_solution, sess, idx)
            if avg_rewards < self.avg_rewards: 
                sample_new, objective_function_new = self.update_teacher(iteration, sess, avg_rewards, population)
            if objective_function_new < self.objective_function_list_score[idx]:
                self.update_student(population, iteration, sess, idx, sample_new, objective_function_new, new_solution)

        return self.teacher
    

    def objective_function(self, policy_params, sess, index):
        self.policy.set_params(policy_params, sess=sess, index=index)
        paths = self.sampler.obtain_samples(log=False, log_prefix='')
        samples_data = self.sampler_processor.process_samples(paths, log=False, log_prefix='')
        # logging.info("################# %s ##################", index)      
        # logging.info("1 %s",-np.mean(samples_data[1]['rewards']))        
        # for i in range(len(self.objective_function_list_score)):
        #     logging.info(f"{-np.mean(samples_data[i]['rewards'])},{i},{index},{self.objective_function_list_score[i]}")  
        ret = np.array([])
        if index == self.batch_size:
            for i in range(self.batch_size):
                ret = np.concatenate((ret, np.sum(samples_data[i]['rewards'], axis=-1)), axis=-1)
        else:
            for i in range(self.batch_size):
                ret = np.concatenate((ret, np.sum(samples_data[i]['rewards'], axis=-1)), axis=-1)
        avg_reward = np.mean(ret)
        
        return -avg_reward, samples_data[index], -np.mean(samples_data[index]['rewards'])
    
