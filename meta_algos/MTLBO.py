import numpy as np
import logging
from utils import logger
import json
import tensorflow as tf
from .MRLCO import MRLCO
import scipy.stats as stats
class MTLBO(MRLCO):  # Inherit from MRLCO to access its methods
    def __init__(self, population_size, policy, env, sampler, sampler_processor, 
                 batch_size, inner_batch_size, clip_value=0.2, inner_lr=0.1):
            
        super().__init__(meta_batch_size=batch_size, meta_sampler=sampler, 
                         meta_sampler_process=sampler_processor, policy=policy)
        
        self.sess = tf.compat.v1.Session()  
        self.sess.run(tf.compat.v1.global_variables_initializer())  
        self.population_size = population_size
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

        # Initialize self.surr_obj as a list
        # self.surr_obj = [None for _ in range(self.policy.meta_batch_size)]

        # # Define placeholders for PPO update
        # self.old_logits = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None, self.policy.action_dim], name='old_logits')
        # self.advs = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name='advs')
        # self.r = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name='returns')

        # Get actions using the get_actions method
        # actions, _, _ = self.policy.get_actions([np.zeros((batch_size, 1, self.policy.obs_dim)) for _ in range(self.policy.meta_batch_size)])

        # Define the surrogate objective for PPO
        # actions = np.squeeze(actions)

        # The actual computation for the surrogate objective will be done elsewhere (e.g., in a method where you have access to task_id)


    def teacher_phase(self, population, iteration, max_iterations, sess):
        np.random.seed(None)
        logging.info('teacher_phase')
        
        # Identify the teacher (best solution in the population)
        if len(self.objective_function_list_score) != len(population):
            for idx, student in enumerate(population):
                score, sample = self.objective_function(student, sess)
                self.objective_function_list_score.append(score)
                self.objective_function_list_sample.append(sample)
        
        max_value, max_index = max((value, index) for index, value in enumerate(self.objective_function_list_score))
        print(len(self.objective_function_list_score), max_value, max_index)
        self.teacher = population[max_index]
        self.teacher_reward = max_value
        self.teacher_sample = self.objective_function_list_sample [max_index]


        logging.info('teacher_phase, teacher')
        
        # Compute the mean solution
        mean_solution = {}
        keys = population[0].keys()
        
        for key in keys:
            mean_solution[key] = np.mean([individual[key] for individual in population], axis=0)
        mean_solution_val, _ = self.objective_function(mean_solution, sess)

        # Calculate w based on the current iteration
        w_start = 0.9  # or another value you choose
        w_end = 0.02    # or another value you choose
        w = w_start - (w_start - w_end) * (iteration / max_iterations)

        # Update the solutions using sine and cosine functions
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
                    self.scale_dict(diff, w * teaching_factor * rand_num * np.cos(angle)),
                    session=sess
                )

            new_solution = self.add_dicts(student, scaled_diff, session=sess)
            objective_function_new, sample_new = self.objective_function(new_solution, sess)
            if objective_function_new > self.objective_function_list_score[idx]:
                population[idx] = new_solution
                logging.info(f'teacher_phase{idx} -- {objective_function_new} -- {self.objective_function_list_score[idx]} -- {self.teacher_reward}')
                self.objective_function_list_score[idx] = objective_function_new
                self.objective_function_list_sample[idx] = sample_new
        return self.teacher_sample





    def subtract_dicts(self,dict1, dict2):
        print('subtract_dicts')
        result = {}
        for key in dict1:
            result[key] = dict1[key] - dict2[key]
        return result

    def scale_dict(self, dict1, scalar):
        print('scale_dict')
        result = {}
        for key in dict1:
            result[key] = dict1[key] * scalar
        return result
    def add_dicts(self, dict1, dict2):
        print('add_dicts')
        result = {}
        for key in dict1:
            result[key] = dict1[key] + dict2[key]
        return result
    # def add_dicts(self, dict1, dict2, session):
    #     result = {}
    #     for key in dict1:
    #         # Check if values are numpy arrays
    #         if isinstance(dict1[key], np.ndarray) and isinstance(dict2[key], np.ndarray):
    #             # Convert numpy arrays to tensors
    #             tensor1 = tf.constant(dict1[key])
    #             tensor2 = tf.constant(dict2[key])
                
    #             # Element-wise addition for tensors
    #             added_tensor = tf.add(tensor1, tensor2)
                
    #             # Convert result tensor back to numpy array using a session and store in result dictionary
    #             result[key] = session.run(added_tensor)
    #         else:
    #             raise ValueError(f"Values for key {key} are not numpy arrays.")
    #     return result


    def learner_phase(self, population, iteration, max_iterations, sess):
        np.random.seed(None)

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
            
            if self.objective_function(student, sess) < self.objective_function(other_learner, sess):
                scaled_diff = self.scale_dict(diff, rand_num)
            else:
                scaled_diff = self.scale_dict(diff, rand_num * np.cos(angle))
            
            new_solution = self.add_dicts(student, scaled_diff, session=sess)
            objective_function_new , sample_new= self.objective_function(new_solution, sess)
            if objective_function_new > self.objective_function_list_score[idx]:
                population[idx] = new_solution
                logging.info(f'First Group learner_phase{idx} -- {objective_function_new} -- {self.objective_function_list_score[idx]} -- {self.teacher_reward}')
                self.objective_function_list_score[idx] = objective_function_new
                self.objective_function_list_sample[idx] = sample_new

        
        # Second Group
        # teacher = min(population, key=self.objective_function)
        
        for idx in range(half_n, n):  # Only considering the second half of the population
            student = population[idx]
            
            diff = self.subtract_dicts(student, self.teacher)
            
            rand_num = np.random.random()
            
            # Adjusting the behavior based on the current iteration
            angle = (np.pi / 2) * (iteration / max_iterations)

            scaled_diff = self.scale_dict(diff, np.cos(angle))
            
            new_solution = self.add_dicts(student, scaled_diff, session=sess)
            
            objective_function_new, sample_new = self.objective_function(new_solution, sess)
            if objective_function_new > self.objective_function_list_score[idx]:
                population[idx] = new_solution
                logging.info(f'Second Group learner_phase{idx} -- {objective_function_new} -- {self.objective_function_list_score[idx]} -- {self.teacher_reward}')
                self.objective_function_list_score[idx] = objective_function_new
                self.objective_function_list_sample[idx] = sample_new
        # paths = self.sampler.obtain_samples(log=False, log_prefix='')
        # samples_data = self.sampler_processor.process_samples(paths, log=False, log_prefix='')
        # # Perform PPO update
        # self.UpdatePPOTarget(samples_data, batch_size=self.inner_batch_size)  # Use the inherited method
       

    # def objective_function(self, policy_params, PPO_check=False):
    #     # Convert policy_params to a string
    #     key = json.dumps(policy_params, sort_keys=True, default=str)

    #     # Check if the result is already in the cache
    #     if key in self.history:
    #         return self.history[key][0], self.history[key][1]

    #     # If not, compute the result
    #     logging.debug('objective_function')
    #     self.policy.set_params(policy_params)
    #     paths = self.sampler.obtain_samples(log=False, log_prefix='')
    #     samples_data = self.sampler_processor.process_samples(paths, log=False, log_prefix='')

    #     all_rewards = [data['rewards'] for data in samples_data]
    #     rewards = np.concatenate(all_rewards, axis=0)
    #     # Compute various metrics
    #     mean_reward = np.mean(rewards)
    #     variance = np.var(rewards)
    #     median_reward = np.median(rewards)
    #     skewness_reward = stats.skew(rewards)
        
    #     # Combine metrics for the objective function
    #     # Here, you can assign different coefficients to each metric based on their importance
    #     combined_metric = mean_reward - self.variance_coefficient * variance + 0.1 * median_reward - 0.05 * skewness_reward
    #     combined_metric = mean_reward 
        
    #     # Store the result in the cache
    #     self.history[key] = [np.mean(combined_metric) , samples_data]
    #     logging.info(f'objective_function  {np.mean(combined_metric)} -- {self.teacher_reward}')
    #     return np.mean(combined_metric), samples_data

    def objective_function(self, policy_params, sess):
        with tf.device('/device:XLA_GPU:0'):
            
                # Convert policy_params to a string
                # key = json.dumps(policy_params, sort_keys=True, default=str)

                # Check if the result is already in the cache
                # if key in self.history:
                #     return self.history[key][0], self.history[key][1]

                # If not, compute the result
                logging.info('objective_function ')
                self.policy.set_params(policy_params, sess=sess)
                logging.info('objective_function set_params')
                paths = self.sampler.obtain_samples(log=False, log_prefix='')
                logging.info('objective_function paths')
                samples_data = self.sampler_processor.process_samples(paths, log=False, log_prefix='')
                logging.info('objective_function samples_data')

                all_rewards = [data['rewards'] for data in samples_data]
                logging.info('objective_function all_rewards')
                
                rewards_placeholder = tf.placeholder(tf.float32, shape=[None, 20])
                mean_reward = tf.reduce_mean(rewards_placeholder)
                variance = tf.math.reduce_variance(rewards_placeholder)
                logging.info('objective_function variance')
                # For median and skewness, you might need to implement custom TensorFlow operations or use an external library.

                # Combine metrics for the objective function
                combined_metric = mean_reward - self.variance_coefficient * variance
                # Add other metrics as needed...

                with tf.Session() as sess:
                    combined_metric_val = sess.run(combined_metric, feed_dict={rewards_placeholder: np.concatenate(all_rewards, axis=0)})
                logging.info('objective_function sess')
                # Store the result in the cache
                # self.history[key] = [combined_metric_val, samples_data]
                logging.info(f'objective_function  {combined_metric_val} -- {self.teacher_reward}')
                print(f'objective_function  {combined_metric_val} -- {self.teacher_reward}')
                return combined_metric_val, samples_data