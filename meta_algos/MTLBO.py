import numpy as np
import logging
from utils import logger
import tensorflow as tf
from .MRLCO import MRLCO
import scipy.stats as stats
import re
import matplotlib.pyplot as plt
import os

class MTLBO():  
    def __init__(self, policy, env, sampler, sampler_processor, 
                 batch_size, inner_batch_size, reward_weight, latency_weight): 
        self.policy = policy
        self.env = env
        self.sampler = sampler
        self.sampler_processor = sampler_processor
        self.variance_coefficient = 0.1
        self.batch_size = batch_size
        self.inner_batch_size = inner_batch_size
        self.reward_weight = reward_weight
        self.latency_weight = latency_weight
        self.teacher_reward = float('inf')  # Changed to 'inf' for better understanding
        self.teacher = []
        self.teacher_sample = []
        self.objective_function_list_score = [None] * batch_size  # Proper initialization to batch size
        self.objective_function_list_sample = [None] * batch_size  # Proper initialization to batch size
        self.avg_rewards = 0
        self.max = batch_size - 1
        self.change = False
        self.result_stat = {}  # Fixed typo from resault_stat to result_stat

        # Initialize logging
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("MTLBO instance is created with batch size: {}".format(batch_size))

        # Check for potential errors in initialization
        if batch_size <= 0:
            raise ValueError("Batch size must be a positive integer")
        if inner_batch_size <= 0:
            raise ValueError("Inner batch size must be a positive integer")
        

    def teacher_phase(self, population, iteration, max_iterations, sess, new_start):
        self.change = False
        logging.info('teacher_phase')
        
        # Calculate initial statistics and set new teacher if new_start is True
        if new_start:
            for idx, student in enumerate(population):
                # Evaluate each student using the objective function
                sample, sample_new, score, avg_latencies = self.objective_function(student, sess, idx)
                # Update the respective lists with the results
                self.objective_function_list_score[idx] = score
                self.objective_function_list_sample[idx] = sample
                # Assume here we calculate initial statistics for all population
            self.result_stat = self.calculate_statistics_tensors(population)
            min_solution_sample = self.result_stat['min']
            # Assume objective_function now also returns latencies
            objective_function_new, sample_new, avg_rewards, avg_latencies = self.objective_function(min_solution_sample, sess=sess, index=self.max)
            self.update_teacher(avg_rewards, avg_latencies, sample_new, objective_function_new, min_solution_sample)

        # Perform teaching phase for each student in the population
        for idx, student in enumerate(population):
            # Calculate teaching factors and perform the teaching strategy
            new_solution, teaching_factor = self.calculate_new_solution(student, idx, iteration, max_iterations)
            
            # Evaluate new solution using the multi-objective approach
            avg_rewards, avg_latencies, sample_new, objective_function_new = self.objective_function(new_solution, sess, self.max)
            self.evaluate_and_update_student(idx, population, avg_rewards, avg_latencies, objective_function_new, sample_new, new_solution, sess)

        return self.teacher
    


    def calculate_new_solution(self, student, idx, iteration, max_iterations):
        # Calculate the mean value of the objective function for comparison
        mean_solution_val = np.mean(self.objective_function_list_score)
        mean_solution_sample = self.result_stat['mean']

        # Calculate the weight for the weighted difference between student and teacher
        w_start = 0.9  # or another value you choose
        w_end = 0.02   # or another value you choose
        w = w_start - (w_start - w_end) * (iteration / max_iterations)

        teaching_factor = round(1 + np.random.rand())  # Teaching factor is either 1 or 2

        # Determine the direction to move based on student's performance compared to the mean
        if self.objective_function_list_score[idx] < mean_solution_val:
            # Student is performing better than average, move towards the teacher
            scaled_diff = self.scale_dict(student, w)  # Scale the student's parameters
            teacher_diff = self.subtract_dicts(self.teacher, student)  # Calculate the difference from the teacher
            teacher_diff = self.scale_dict(teacher_diff, np.random.random())  # Scale the difference randomly
            new_solution = self.add_dicts(scaled_diff, teacher_diff)  # Create a new solution by adding the scaled differences
        else:
            # Student is performing worse than average, try a new direction
            tf_mean = self.scale_dict(mean_solution_sample, teaching_factor)  # Scale the global mean parameters
            diff = self.subtract_dicts(self.teacher, tf_mean)  # Calculate the difference from the teacher to mean
            rand_num = 2 * (np.random.random() - 0.5)  # A random number between -1 and 1
            angle = (np.pi / 2) * (iteration / max_iterations)  # Angle for trigonometric scaling

            # Create new solution vectors based on trigonometric transformations
            x_new_first = self.scale_dict(self.add_dicts(student, self.scale_dict(self.subtract_dicts(mean_solution_sample, student), rand_num)), np.sin(angle))
            x_new_second = self.scale_dict(diff, np.cos(angle))
            new_solution = self.add_dicts(x_new_first, x_new_second)  # Combine the trigonometrically scaled vectors

        # Return the new solution and the teaching factor used in its generation
        return new_solution, teaching_factor

    def evaluate_and_update_student(self, population, idx, avg_rewards, avg_latencies, objective_function_new, sample_new, new_solution, sess):
        # Calculate the weighted objective which might consider multiple factors like rewards and latencies
        weighted_objective_new = self.calculate_weighted_objective(avg_rewards, avg_latencies)


        # Compare and update if the new solution's weighted objective is better than the current one
        if weighted_objective_new < self.objective_function_list_score[idx]:
            self.change = True  # Flag to indicate a change has occurred
            self.objective_function_list_score[idx] = weighted_objective_new  # Update the score list
            self.objective_function_list_sample[idx] = sample_new  # Update the sample list
            population[idx] = new_solution  # Update the student in the population

            # Log the update
            logging.info(f'Updated student {idx} with new weighted objective: {weighted_objective_new}')
            
            # Update the policy parameters with the new solution
            self.policy.set_params(new_solution, sess=sess, index=idx)

        # Return the possibly updated population
        return population


    def calculate_weighted_objective(self, rewards, latencies):
        print(rewards, type(rewards),latencies , type(latencies))
        # Convert rewards and latencies to NumPy arrays if they are not already arrays
        rewards_array = np.array(rewards) if not isinstance(rewards, float) else rewards
        latencies_array = np.array(latencies) if not isinstance(latencies, float) else latencies

        # Now perform the weighted calculation
        weighted_objective = (self.reward_weight * rewards_array) - (self.latency_weight * latencies_array)
        return weighted_objective

    def update_teacher(self, avg_rewards, avg_latencies, sample_new, objective_function_new, solution_sample):
        weighted_objective_new = self.calculate_weighted_objective(avg_rewards, avg_latencies)
        
        # Update teacher if the new solution has a better weighted objective
        if weighted_objective_new < self.teacher_reward:
            self.change = True
            self.teacher_reward = weighted_objective_new
            self.teacher_sample = sample_new
            self.teacher = solution_sample
            logging.info(f'Updated teacher with new weighted objective: {weighted_objective_new}')



    def calculate_statistics_tensors(self, population):
        mean_solution_sample = {}
        max_solution_sample = {}
        min_solution_sample = {}
        stats_solution_sample = {}
        all_keys = set()
        for pop in population:
            all_keys.update(pop.keys())

        base_keys = set([re.sub(r'task_[0-9]{1,2}_policy/', '', key) for key in all_keys])

        for base_key in base_keys:
            matching_keys = [k for k in all_keys if re.sub(r'task_[0-9]{1,2}_policy/', '', k) == base_key]
            values_list = []
            for pop in population:
                for key in matching_keys:
                    if key in pop and pop[key] is not None:
                        values_list.append(pop[key])
            
            if values_list:  # Proceed if values_list is not empty
                if isinstance(values_list[0], (list, np.ndarray)):  # Check if the values are list or ndarray
                    mean_values = np.mean(values_list, axis=0).tolist()
                    max_values = np.max(values_list, axis=0).tolist()
                    min_values = np.min(values_list, axis=0).tolist()
                else:  # If values are scalar
                    mean_values = np.mean(values_list)
                    max_values = np.max(values_list)
                    min_values = np.min(values_list)

                mean_solution_sample[base_key] = mean_values
                max_solution_sample[base_key] = max_values
                min_solution_sample[base_key] = min_values
            else:  # Handle the empty case
                logging.warning(f"No values to calculate statistics for key: {base_key}")
                continue  # Skip this key or set default values as needed

        stats_solution_sample['mean'] = mean_solution_sample
        stats_solution_sample['max'] = max_solution_sample
        stats_solution_sample['min'] = min_solution_sample

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
        sorted_indices = sorted(range(len(self.objective_function_list_score)), key=lambda i: self.objective_function_list_score[i])

        # Split the sorted indices into two halves
        midpoint = len(sorted_indices) // 2
        above_average = sorted_indices[:midpoint]
        below_average = sorted_indices[midpoint:]

        if not above_average or not below_average:
            logging.error("Population does not have a distinct above and below average split.")
            return None  # Changed from os.exit(1) for better error handling in most cases.

        # Iterate through below-average students to update based on the learner phase logic
        for idx in below_average:
            student = population[idx]
            j = np.random.choice(above_average)  # Choosing from above_average for comparison
            other_learner = population[j]

            # Calculate new solution based on the learning strategy
            new_solution, _ = self.calculate_new_solution(student, idx, iteration, max_iterations)
            
            # Evaluate new solution using multi-objective function
            avg_rewards, avg_latencies, objective_function_new, sample_new = self.multi_objective_function(new_solution, sess, idx)
            self.evaluate_and_update_student(idx, avg_rewards, avg_latencies, objective_function_new, sample_new, new_solution, sess, population)

        # Iterate through above-average students to refine their solutions
        for idx in above_average:
            student = population[idx]
            new_solution, _ = self.calculate_new_solution(student, idx, iteration, max_iterations)
            
            # Evaluate new solution using multi-objective function
            avg_rewards, avg_latencies, objective_function_new, sample_new = self.multi_objective_function(new_solution, sess, idx)
            self.evaluate_and_update_student(idx, avg_rewards, avg_latencies, objective_function_new, sample_new, new_solution, sess, population)

        return self.teacher

    def objective_function(self, policy_params, sess, index, reward_weight=0.5, latency_weight=0.5):
        """
        Calculate the objective function considering multiple objectives.
        This function assumes that higher rewards and lower latencies are better.

        :param policy_params: Parameters of the policy to evaluate.
        :param sess: TensorFlow session.
        :param index: Index of the current evaluation.
        :param reward_weight: Weight for the reward objective.
        :param latency_weight: Weight for the latency objective.
        :return: Scalarized objective score, the samples data, and the scalarized metric components.
        """
        # Set policy parameters and collect the required samples
        self.policy.set_params(policy_params, sess=sess, index=index)
        paths = self.sampler.obtain_samples(log=False, log_prefix='')
        samples_data = self.sampler_processor.process_samples(paths, log="all", log_prefix='')
        
        # Compute reward and latency
        rewards = -np.mean([np.sum(path_data['rewards']) for path_data in samples_data])  # Assuming rewards should be maximized
        latencies = np.mean([path_data['finish_time'] for path_data in samples_data if 'finish_time' in path_data])  # Assuming latency data is available and should be minimized
        
        # Normalize latencies if required
        # normalized_latency = (latencies - np.min(latencies)) / (np.max(latencies) - np.min(latencies))
        
        # Calculate the weighted sum of normalized objectives
        objective_score = reward_weight * rewards + latency_weight * (1/latencies)  # Assuming that lower latency is better and needs to be inverted
        return objective_score, samples_data, rewards, latencies
     

