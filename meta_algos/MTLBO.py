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
        self.objective_function_list_score=list(range(self.batch_size-1))
        self.objective_function_list_sample=list(range(self.batch_size-1))
        self.avg_rewards=9999999
        self.max=batch_size-1
        self.change = False
        self.resault_stat ={}
        self.w_start = 0.9  
        self.w_end = 0.01
        outer_lr=1e-4
        inner_lr=0.1
        num_inner_grad_steps=4
        clip_value = 0.2
        vf_coef=0.5
        max_grad_norm=0.5 
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.num_inner_grad_steps=num_inner_grad_steps
        self.policy = policy
        self.meta_sampler = sampler
        self.meta_sampler_process = sampler_processor
        self.meta_batch_size = batch_size
        self.update_numbers = 1

        #self.optimizer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=self.lr, epsilon=1e-5)
        #self.inner_optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.inner_lr)
        self.inner_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.inner_lr)
        self.outer_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.outer_lr)
        self.clip_value = clip_value
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        # initialize the place hoder for each task place holder.
        self.new_logits = []
        self.decoder_inputs =[]
        self.old_logits = []
        self.actions = []
        self.obs = []
        self.vpred = []
        self.decoder_full_length = []

        self.old_v =[]
        self.advs = []
        self.r = []

        self.surr_obj = []
        self.vf_loss = []
        self.total_loss = []
        self._train = []

        self.build_graph()

    def build_graph(self):
        # build inner update for each tasks
        for i in range(self.meta_batch_size):
            self.new_logits.append(self.policy.meta_policies[i].network.decoder_logits)
            self.decoder_inputs.append(self.policy.meta_policies[i].decoder_inputs)
            self.old_logits.append(tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None, self.policy.action_dim], name='old_logits_ph_task_'+str(i)))
            self.actions.append(self.policy.meta_policies[i].decoder_targets)
            self.obs.append(self.policy.meta_policies[i].obs)
            self.vpred.append(self.policy.meta_policies[i].vf)
            self.decoder_full_length.append(self.policy.meta_policies[i].decoder_full_length)

            self.old_v.append(tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name='old_v_ph_task_'+str(i)))
            self.advs.append(tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name='advs_ph_task'+str(i)))
            self.r.append(tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name='r_ph_task_'+str(i)))

            with tf.compat.v1.variable_scope("inner_update_parameters_task_"+str(i)) as scope:
                likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(self.actions[i], self.old_logits[i], self.new_logits[i])

                clipped_obj = tf.minimum(likelihood_ratio * self.advs[i] ,
                                         tf.clip_by_value(likelihood_ratio,
                                                          1.0 - self.clip_value,
                                                          1.0 + self.clip_value) * self.advs[i])
                self.surr_obj.append(-tf.reduce_mean(clipped_obj))

                vpredclipped = self.vpred[i] + tf.clip_by_value(self.vpred[i] - self.old_v[i], -self.clip_value, self.clip_value)
                vf_losses1 = tf.square(self.vpred[i] - self.r[i])
                vf_losses2 = tf.square(vpredclipped - self.r[i])

                self.vf_loss.append( .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2)) )

                self.total_loss.append( self.surr_obj[i] + self.vf_coef * self.vf_loss[i])

                params = self.policy.meta_policies[i].network.get_trainable_variables()

                grads_and_var = self.inner_optimizer.compute_gradients(self.total_loss[i], params)
                grads, var = zip(*grads_and_var)

                if self.max_grad_norm is not None:
                    # Clip the gradients (normalize)
                    grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                grads_and_var = list(zip(grads, var))

                self._train.append(self.inner_optimizer.apply_gradients(grads_and_var))

        # Outer update for the parameters
        # feed in the parameters of inner policy network and update outer parameters.
        with tf.compat.v1.variable_scope("outer_update_parameters") as scope:
            core_network_parameters = self.policy.core_policy.get_trainable_variables()
            self.grads_placeholders = []

            for i, var in enumerate(core_network_parameters):
                self.grads_placeholders.append(tf.compat.v1.placeholder(shape=var.shape, dtype=var.dtype, name="grads_"+str(i)))

            outer_grads_and_var = list(zip(self.grads_placeholders, core_network_parameters))

            self._outer_train = self.outer_optimizer.apply_gradients(outer_grads_and_var)
        
    def teacher_phase(self, population, iteration, max_iterations, sess, new_start):
        self.change = False
        logging.info('teacher_phase')
        if new_start:        
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
            # logging.info(f'teacher_phase subtract_dicts{iteration}')

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
            
            avg_rewards, sample_new, objective_function_new = self.objective_function(new_solution, sess, self.max)
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
        avg_rewards, sample_new, objective_function_new = self.objective_function(solution_sample, sess=sess, index=self.max)
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
                    weighted_teacher[key] = 0  # Initialize if not present
                # Add the weighted value for each key
                weight = self.get_weight_for_policy(index) 
                weighted_teacher[key] += (weight * np.array(policy[key]))
        return weighted_teacher
    

    def get_weight_for_policy(self, policy_index):
        if not isinstance(policy_index, int):
            raise TypeError("policy_index must be an integer")
        performance_metric = self.objective_function_list_score[policy_index]
        weight = performance_metric / (sum(self.objective_function_list_score))  # Adding 1 to avoid division by zero
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
                min_solution_sample , max_solution_sample= self.calculate_bounds_separate(population)
                rand_num = (np.random.random() -0.5) * 2
                diff = self.subtract_dicts(max_solution_sample, min_solution_sample)
                scaled_diff = self.scale_dict(diff, rand_num * np.cos(angle))
            
            new_solution = scaled_diff
            avg_rewards, sample_new, objective_function_new = self.objective_function(new_solution, sess, self.max)
            if avg_rewards < self.avg_rewards:
                sample_new, objective_function_new = self.update_teacher(iteration, sess, avg_rewards, population)
            if objective_function_new < self.objective_function_list_score[idx]:
                self.update_student(population, iteration, sess, idx, sample_new, objective_function_new, new_solution)

        for idx in above_average:  
            student = population[idx]
            diff = self.subtract_dicts(self.teacher, student)
            rand_num = np.random.random()            
            angle = (np.pi / 2) * (iteration / max_iterations)
            scaled_diff = self.scale_dict(diff, np.cos(angle))
            new_solution = self.add_dicts(student, scaled_diff)
            avg_rewards, sample_new, objective_function_new = self.objective_function(new_solution, sess, self.max)
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
        if index == self.batch_size-1:
            for i in range(len(samples_data)):
                ret = np.concatenate((ret, np.sum(samples_data[i]['rewards'], axis=-1)), axis=-1)
        else:
            for i in range(len(samples_data)-1):
                ret = np.concatenate((ret, np.sum(samples_data[i]['rewards'], axis=-1)), axis=-1)
        avg_reward = np.mean(ret)
        
        return -avg_reward, samples_data[index], -np.mean(samples_data[index]['rewards'])
    
    def UpdatePPOTarget(self, task_samples, batch_size=50):
        total_policy_losses = []
        total_value_losses = []
        for i in range(self.meta_batch_size):
            policy_losses, value_losses = self.UpdatePPOTargetPerTask(task_samples[i], i, batch_size)
            total_policy_losses.append(policy_losses)
            total_value_losses.append(value_losses)

        return total_policy_losses, total_value_losses

    def UpdatePPOTargetPerTask(self, task_samples, task_id, batch_size=50):
        policy_losses = []
        value_losses = []

        batch_number = int(task_samples['observations'].shape[0] / batch_size)
        self.update_numbers = batch_number
        #:q!
        # print("update number is: ", self.update_numbers)
        #observations = task_samples['observations']

        shift_actions = np.column_stack(
                    (np.zeros(task_samples['actions'].shape[0], dtype=np.int32), task_samples['actions'][:, 0:-1]))

        observations_batchs = np.split(np.array(task_samples['observations']), batch_number)
        actions_batchs = np.split(np.array(task_samples['actions']), batch_number)
        shift_action_batchs = np.split(np.array(shift_actions), batch_number)

        old_logits_batchs = np.split(np.array(task_samples["logits"], dtype=np.float32 ), batch_number)
        advs_batchs = np.split(np.array(task_samples['advantages'], dtype=np.float32), batch_number)
        oldvpred = np.split(np.array(task_samples['values'], dtype=np.float32), batch_number)
        returns = np.split(np.array(task_samples['returns'], dtype=np.float32), batch_number)

        sess = tf.compat.v1.get_default_session()

        vf_loss = 0.0
        pg_loss = 0.0
        # copy_policy.set_weights(self.policy.get_weights())
        for i in range(self.num_inner_grad_steps):
            # action, old_logits, _ = copy_policy(observations)
            for old_logits, old_v, observations, actions, shift_actions, advs, r in zip(old_logits_batchs, oldvpred, observations_batchs, actions_batchs,
                                                                                        shift_action_batchs, advs_batchs, returns):
                decoder_full_length = np.array([observations.shape[1]] * observations.shape[0], dtype=np.int32)

                feed_dict = {self.old_logits[task_id]: old_logits, self.old_v[task_id]: old_v, self.obs[task_id]: observations, self.actions[task_id]: actions,
                            self.decoder_inputs[task_id]: shift_actions,
                             self.decoder_full_length[task_id]: decoder_full_length, self.advs[task_id]: advs, self.r[task_id]: r}

                _, value_loss, policy_loss = sess.run([self._train[task_id], self.vf_loss[task_id], self.surr_obj[task_id]], feed_dict=feed_dict)

                vf_loss += value_loss
                pg_loss += policy_loss

            vf_loss = vf_loss / float(self.num_inner_grad_steps)
            pg_loss = pg_loss / float(self.num_inner_grad_steps)

            value_losses.append(vf_loss)
            policy_losses.append(pg_loss)

        return policy_losses, value_losses