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
        self.teacher_reward = 999999
        self.teacher = []
        self.teacher_sample = []
        self.objective_function_list_score=list(range(self.batch_size-1))
        self.objective_function_list_sample=list(range(self.batch_size-1))
        self.avg_rewards=0

    def teacher_phase(self, population, iteration, max_iterations, sess, population_index):
        logging.info('teacher_phase')        
        # if len(self.objective_function_list_score) != len(population):
        for idx, student in enumerate(population):
            self.avg_rewards, sample, score = self.objective_function(student, sess, population_index[idx])
            self.objective_function_list_score[idx]=score
            self.objective_function_list_sample[idx]=sample
        # self.teacher = population[0]
        # self.teacher_reward = self.objective_function_list_score[0]
        # self.teacher_sample = self.objective_function_list_sample [0]
            if   score < self.teacher_reward:
                self.teacher = student
                self.teacher_reward = score
                self.teacher_sample = sample
                logging.info(f'teacher_phase{idx} Teacher changed{self.teacher_reward}')
        
        # max_value, max_index = max((value, index) for index, value in enumerate(self.objective_function_list_score))
        # print(len(self.objective_function_list_score), max_value, max_index)
        # self.teacher = population[0]
        # self.teacher_reward = max_value
        # self.teacher_sample = self.objective_function_list_sample [max_index]
        
        mean_solution_val = np.mean(self.objective_function_list_score)

        mean_solution_sample = {}

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
            else:
                mean_values = np.mean(values_list)
            
            mean_solution_sample[base_key] = mean_values

            # Log the result
            # logging.info(f"Key: {base_key}, {np.array(mean_solution_sample[base_key]).shape}")




        w_start = 0.9  # or another value you choose
        w_end = 0.02    # or another value you choose
        w = w_start - (w_start - w_end) * (iteration / max_iterations)
        # _, sample_mean, _ = self.objective_function(mean_solution_sample, sess, 0)
        for idx, student in enumerate(population):
            teaching_factor = round(1 + np.random.rand())
            logging.info(f'teacher_phase subtract_dicts{idx}')
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
            
            
            avg_rewards, sample_new, objective_function_new = self.objective_function(new_solution, sess, 4)
            # print(objective_function_new,self.objective_function_list_score[idx],idx)
            if avg_rewards < self.avg_rewards:
                    population[idx] = new_solution
                    # logging.info(f'teacher_phase{idx} -- {objective_function_new} -- {self.objective_function_list_score[idx]} -- {self.teacher_reward}')
                    self.objective_function_list_score[idx] = objective_function_new
                    self.objective_function_list_sample[idx] = sample_new
                    self.avg_rewards = avg_rewards
                    # logging.info("set_params %s", idx)
                    self.objective_function(new_solution, sess=sess, index=idx)
            elif objective_function_new < self.objective_function_list_score[idx]:
                    population[idx] = new_solution
                    # logging.info(f'teacher_phase{idx} -- {objective_function_new} -- {self.objective_function_list_score[idx]} -- {self.teacher_reward}')
                    self.objective_function_list_score[idx] = objective_function_new
                    self.objective_function_list_sample[idx] = sample_new
                    # logging.info("set_params %s", idx)
                    self.objective_function(new_solution, sess=sess, index=idx)
            if  self.objective_function_list_score[idx] < self.teacher_reward:
                self.teacher = population[idx]
                self.teacher_reward = self.objective_function_list_score[idx]
                self.teacher_sample = self.objective_function_list_sample [idx]
                logging.info(f'teacher_phase{idx} Teacher changed{self.teacher_reward}')

        return self.teacher_sample

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


    def learner_phase(self, population, iteration, max_iterations, sess, population_index):
        logging.info('learner_phase')
        n = len(population)
        half_n = n // 2        
        for idx in range(half_n):  
            student = population[idx]
            j = np.random.choice([x for x in range(half_n) if x != idx])
            other_learner = population[j]
            diff = self.subtract_dicts(other_learner, student)
            rand_num = np.random.random()            
            angle = (np.pi / 2) * (iteration / max_iterations)
            
            if self.objective_function_list_score[idx] < self.objective_function_list_score[j]:
                scaled_diff = self.add_dicts(diff, self.scale_dict(diff, np.cos(angle)))
            else:
                scaled_diff = self.scale_dict(diff, rand_num * np.cos(angle))
            
            new_solution = scaled_diff
            avg_rewards, sample_new, objective_function_new = self.objective_function(new_solution, sess, 4)
            # print(objective_function_new,self.objective_function_list_score[idx],idx)
            if avg_rewards < self.avg_rewards:
                    population[idx] = new_solution
                    logging.info(f'learner_phase{idx} -- {objective_function_new} -- {self.objective_function_list_score[idx]} -- {self.teacher_reward}')
                    self.objective_function_list_score[idx] = objective_function_new
                    self.objective_function_list_sample[idx] = sample_new
                    self.avg_rewards = avg_rewards
                    self.objective_function(new_solution, sess=sess, index=idx)
                    logging.info(f'teacher_phase{idx} Teacher changed')
            elif objective_function_new < self.objective_function_list_score[idx]:
                    population[idx] = new_solution
                    logging.info(f'learner_phase{idx} -- {objective_function_new} -- {self.objective_function_list_score[idx]} -- {self.teacher_reward}')
                    self.objective_function_list_score[idx] = objective_function_new
                    self.objective_function_list_sample[idx] = sample_new
                    self.objective_function(new_solution, sess=sess, index=idx)
            if   self.objective_function_list_score[idx] < self.teacher_reward:
                self.teacher = population[idx]
                self.teacher_reward = self.objective_function_list_score[idx]
                self.teacher_sample = self.objective_function_list_sample [idx]
                logging.info(f'teacher_phase{idx} Teacher changed{self.teacher_reward}')

        for idx in range(half_n, n):  
            student = population[idx]
            diff = self.subtract_dicts(self.teacher, student)
            rand_num = np.random.random()            
            angle = (np.pi / 2) * (iteration / max_iterations)
            scaled_diff = self.scale_dict(diff, np.cos(angle))
            new_solution = self.add_dicts(student, scaled_diff)
            avg_rewards, sample_new, objective_function_new = self.objective_function(new_solution, sess, 4)
            # print(objective_function_new,self.objective_function_list_score[idx],idx)
            if avg_rewards < self.avg_rewards:
                    population[idx] = new_solution
                    logging.info(f'teacher_phase{idx} -- {objective_function_new} -- {self.objective_function_list_score[idx]} -- {self.teacher_reward}')
                    self.objective_function_list_score[idx] = objective_function_new
                    self.objective_function_list_sample[idx] = sample_new
                    self.avg_rewards = avg_rewards
                    self.objective_function(new_solution, sess=sess, index=idx)
            elif objective_function_new < self.objective_function_list_score[idx]:
                    population[idx] = new_solution
                    logging.info(f'teacher_phase{idx} -- {objective_function_new} -- {self.objective_function_list_score[idx]} -- {self.teacher_reward}')
                    self.objective_function_list_score[idx] = objective_function_new
                    self.objective_function_list_sample[idx] = sample_new
                    self.objective_function(new_solution, sess=sess, index=idx)
            if   self.objective_function_list_score[idx] < self.teacher_reward:
                self.teacher = population[idx]
                self.teacher_reward = self.objective_function_list_score[idx]
                self.teacher_sample = self.objective_function_list_sample [idx]
                logging.info(f'teacher_phase{idx} Teacher changed{self.teacher_reward}')


    # def objective_function(self, policy_params, sess, index):
    #     # print('objective_function')
    #     self.policy.set_params(policy_params, sess=sess, index=index)
    #     # print('objective_function set_params')    
    #     paths = self.sampler.obtain_samples(log=False, log_prefix='')
    #     samples_data = self.sampler_processor.process_samples(paths, log=False, log_prefix='')
    #     # all_rewards = np.concatenate([data['rewards'] for data in samples_data], axis=0)
    #     # mean_reward = np.mean(all_rewards)
        
    #     # variance = np.var(all_rewards)
    #     # combined_metric = mean_reward - self.variance_coefficient * variance
    #     # ret = np.array([])
    #     # for i in range(0,5):
    #     #     ret = np.concatenate((ret, np.linalg.norm(samples_data[i]['rewards'], axis=-1)), axis=-1)
    #     # for i in range(0,5):
    #     #     print("linalg","sum","avg")
    #     #     print(np.linalg.norm(samples_data[i]['rewards'], axis=-1) ,np.sum(samples_data[i]['rewards'], axis=-1),np.mean(samples_data[i]['rewards'], axis=-1))
             

    #     # avg_reward = np.mean(ret)
    #     # # avg_reward = np.linalg.norm(samples_data['rewards'])
    #     # # avg_reward=0
    #     # # for i in range(1, 5):
    #     # #     avg_reward = avg_reward +np.sum(samples_data[i]['rewards'])
    #     # return avg_reward, samples_data[index], np.linalg.norm(samples_data[index]['rewards'], axis=-1)
    


    #     ### momentom 



    #         # List to store mean rewards and variances for each dataset
    #     mean_rewards = []
    #     variances = []
        
    #     # Loop over each dataset (assuming each dataset is represented by a 'data' in samples_data)
    #     for data in samples_data:
    #         rewards = data['rewards']
    #         mean_rewards.append(np.mean(rewards))
    #         variances.append(np.var(rewards))            
    #     # Compute global mean reward and global mean variance
    #     combined_metric = np.mean(mean_rewards)
    #     global_mean_variance = np.mean(variances)
        
    #     # Combined metric considering global mean reward and global mean variance
    #     # combined_metric = global_mean_reward - self.variance_coefficient * global_mean_variance
        
    #     return -combined_metric, samples_data[index], -np.mean(samples_data[index]['rewards'])
    def objective_function(self, policy_params, sess, index):
        self.policy.set_params(policy_params, sess=sess, index=index)
        paths = self.sampler.obtain_samples(log=False, log_prefix='')
        samples_data = self.sampler_processor.process_samples(paths, log=False, log_prefix='')

        # List to store mean rewards and variances for each dataset
        mean_rewards = []
        variances = []
        
        # Define weights for datasets (assuming equal importance for now)
        weights = [1.0 for _ in samples_data]

        # Loop over each dataset
        for data, weight in zip(samples_data, weights):
            rewards = data['rewards']
            # mean_rewards.append(np.mean(rewards) * weight)
            mean_rewards.append(np.mean(rewards))
            variances.append(np.var(rewards))


        # Compute global mean reward and global mean variance
        global_mean_reward = np.mean(mean_rewards)
        global_mean_variance = np.mean(variances)

        # Introduce a penalty term for extreme variances
        variance_penalty = np.sum([max(0, var - global_mean_variance) for var in variances])

        # Compute the momentum term (difference in rewards between consecutive iterations)
        # For simplicity, let's assume the previous rewards are stored in a list called self.prev_rewards
        if hasattr(self, 'prev_rewards'):
            momentum_term = np.mean(mean_rewards) - np.mean(self.prev_rewards)
        else:
            momentum_term = 0

        # Update the previous rewards
        self.prev_rewards = mean_rewards

        # Combined metric considering global mean reward, variance penalty, and momentum
        combined_metric = global_mean_reward - self.variance_coefficient * variance_penalty + momentum_term
        logging.info("################# %s ##################", index)      
        logging.info("4 %s",-np.mean(samples_data[4]['rewards']))        
        for i in range(len(self.objective_function_list_score)):
            logging.info(f"{-np.mean(samples_data[i]['rewards'])},{i},{index},{self.objective_function_list_score[i]}")  
 
        logging.info("################# %s  teacher %s ##################",-combined_metric, self.teacher_reward)  
        # plt.figure(figsize=(14, 10))

        # for i in range(len(samples_data)):
        #     plt.plot(samples_data[i]['rewards'], label=f'Set {i}')
        # plt.title('Rewards for 5 Sets')
        # plt.xlabel('Reward Index')
        # plt.ylabel('Reward Value')
        # plt.legend()
        # plt.grid(True)
        # plt.subplots_adjust(bottom=0.15, top=0.95)  # Adjusting the spacing

        # # Save the plot to a file
        # plt.savefig('plot/rewards_plot.png')  # Modify the path as needed

        # plt.show()
        # plt.close() 
        return -global_mean_reward, samples_data[index], -mean_rewards[index]
