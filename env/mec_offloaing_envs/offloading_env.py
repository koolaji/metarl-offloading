# from env.base import MetaEnv
from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph
import numpy as np
import logging
import time

def _computation_cost(data, processing_power):
    computation_time = data / processing_power

    return computation_time


class Resources(object):
    def __init__(self, mec_process_capable,
                 mobile_process_capable, bandwidth_up=7.0, bandwidth_dl=7.0):
        self.mec_process_capable = mec_process_capable
        self.mobile_process_capable = mobile_process_capable
        self.mobile_process_available_time = 0.0
        self.mec_process_available_time = 0.0

        self.bandwidth_up = bandwidth_up
        self.bandwidth_dl = bandwidth_dl

    def up_transmission_cost(self, data):
        rate = self.bandwidth_up * (1024.0 * 1024.0 / 8.0)

        transmission_time = data / rate

        return transmission_time

    def reset(self):
        self.mec_process_available_time = 0.0
        self.mobile_process_available_time = 0.0

    def dl_transmission_cost(self, data):
        rate = self.bandwidth_dl * (1024.0 * 1024.0 / 8.0)
        transmission_time = data / rate

        return transmission_time

    def locally_execution_cost(self, data):
        return _computation_cost(data, self.mobile_process_capable)

    def mec_execution_cost(self, data):
        return _computation_cost(data, self.mec_process_capable)


class OffloadingEnvironment(object):
    def __init__(self, resource_cluster, batch_size,
                 graph_number,
                 graph_file_paths, time_major):
        logging.debug('Start OffloadingEnvironment')
        current_time = time.time()

        self.resource_cluster = resource_cluster
        self.task_graphs_batches = []
        self.encoder_batches = []
        self.encoder_lengths = []
        self.decoder_full_lengths = []
        self.max_running_time_batches = []
        self.min_running_time_batches = []
        self.graph_file_paths = graph_file_paths
        for graph_file_path in graph_file_paths:
            logging.info('graph_file_path == %s, len(self.encoder_batches), %s',graph_file_path, str(len(self.encoder_batches)))
            encoder_batches, encoder_lengths, task_graph_batches, decoder_full_lengths, max_running_time_batches, \
            min_running_time_batches = self.generate_point_batch_for_random_graphs(batch_size, graph_number,
                                                                                   graph_file_path, time_major)
            self.encoder_batches += encoder_batches
            self.encoder_lengths += encoder_lengths
            self.task_graphs_batches += task_graph_batches
            self.decoder_full_lengths += decoder_full_lengths
            self.max_running_time_batches += max_running_time_batches
            self.min_running_time_batches += min_running_time_batches
        self.total_task = len(self.encoder_batches)
        self.optimal_solution = -1
        self.task_id = -1
        self.time_major = time_major
        self.input_dim = np.array(encoder_batches[0]).shape[-1]
        self.graph_file_paths = graph_file_paths
        self.graph_number = graph_number
        self.local_exe_time = self.get_all_locally_execute_time()
        self.mec_exe_time = self.get_all_mec_execute_time()
        elapsed_time = time.time() - current_time
        logging.debug("Elapsed time: %s", elapsed_time)
        logging.debug('END OffloadingEnvironment')

    def sample_tasks(self, n_tasks):
        logging.info('sample_tasks n_tasks = %s , self.total_task %s',str(n_tasks) ,str(self.total_task))
        return np.random.choice(np.arange(self.total_task), n_tasks, replace=False)
        # return [i for i in range(self.total_task-1)]
    
    def set_task(self, task):
        self.task_id = task

    def get_task(self):
        return self.graph_file_paths[self.task_id]

    def step(self, action):
        logging.debug('Start OffloadingEnvironment step')
        plan_batch = []
        task_graph_batch = self.task_graphs_batches[self.task_id]
        max_running_time_batch = self.max_running_time_batches[self.task_id]
        min_running_time_batch = self.min_running_time_batches[self.task_id]

        for action_sequence, task_graph in zip(action, task_graph_batch):
            plan_sequence = []

            for action, task_id in zip(action_sequence, task_graph.prioritize_sequence):
                plan_sequence.append((task_id, action))

            plan_batch.append(plan_sequence)

        reward_batch, task_finish_time = self.get_reward_batch_step_by_step(plan_batch,
                                                                            task_graph_batch,
                                                                            max_running_time_batch,
                                                                            min_running_time_batch)

        done = True
        observation = np.array(self.encoder_batches[self.task_id])
        info = task_finish_time

        return observation, reward_batch, done, info

    def reset(self):
        logging.debug('Start OffloadingEnvironment reset')
        self.resource_cluster.reset()
        return np.array(self.encoder_batches[self.task_id])

    def generate_point_batch_for_random_graphs(self, batch_size, graph_number, graph_file_path, time_major):
        logging.debug('Start generate_point_batch_for_random_graphs')
        encoder_list = []
        task_graph_list = []

        encoder_batches = []
        encoder_lengths = []
        task_graph_batches = []
        decoder_full_lengths = []

        max_running_time_vector = []
        min_running_time_vector = []

        max_running_time_batches = []
        min_running_time_batches = []

        for i in range(graph_number):
            task_graph = OffloadingTaskGraph(graph_file_path + str(i) + '.gv')
            task_graph_list.append(task_graph)

            max_time, min_time = self.calculate_max_min_runningcost(task_graph.max_data_size,
                                                                    task_graph.min_data_size)
            max_running_time_vector.append(max_time)
            min_running_time_vector.append(min_time)
            scheduling_sequence = task_graph.prioritize_tasks(self.resource_cluster)
            task_encode = np.array(task_graph.encode_point_sequence_with_ranking_and_cost(scheduling_sequence,
                                                                                          self.resource_cluster),
                                   dtype=np.float32)
            encoder_list.append(task_encode)

        for i in range(int(graph_number / batch_size)):
            start_batch_index = i * batch_size
            end_batch_index = (i + 1) * batch_size

            task_encode_batch = encoder_list[start_batch_index:end_batch_index]
            if time_major:
                task_encode_batch = np.array(task_encode_batch).swapaxes(0, 1)
                sequence_length = np.asarray([task_encode_batch.shape[0]] * task_encode_batch.shape[1])
            else:
                task_encode_batch = np.array(task_encode_batch)
                sequence_length = np.asarray([task_encode_batch.shape[1]] * task_encode_batch.shape[0])
                
            decoder_full_lengths.append(sequence_length)
            encoder_lengths.append(sequence_length)
            encoder_batches.append(task_encode_batch)

            task_graph_batch = task_graph_list[start_batch_index:end_batch_index]
            task_graph_batches.append(task_graph_batch)
            max_running_time_batches.append(max_running_time_vector[start_batch_index:end_batch_index])
            min_running_time_batches.append(min_running_time_vector[start_batch_index:end_batch_index])
        logging.debug('END generate_point_batch_for_random_graphs')
        return encoder_batches, encoder_lengths, task_graph_batches, \
               decoder_full_lengths, max_running_time_batches, \
               min_running_time_batches
        
    def calculate_max_min_runningcost(self, max_data_size, min_data_size):
        logging.debug('calculate_max_min_runningcost')
        max_time = max([self.resource_cluster.up_transmission_cost(max_data_size),
                        self.resource_cluster.dl_transmission_cost(max_data_size),
                        self.resource_cluster.locally_execution_cost(max_data_size)])

        min_time = self.resource_cluster.mec_execution_cost(min_data_size)

        return max_time, min_time

    def get_scheduling_cost_step_by_step(self, plan, task_graph):
        # logging.debug('get_scheduling_cost_step_by_step')
        cloud_available_time = 0.0
        ws_available_time = 0.0
        local_available_time = 0.0
        T_l = [0] * task_graph.task_number
        T_ul = [0] * task_graph.task_number
        T_dl = [0] * task_graph.task_number
        FT_cloud = [0] * task_graph.task_number
        FT_ws = [0] * task_graph.task_number
        FT_locally = [0] * task_graph.task_number
        FT_wr = [0] * task_graph.task_number
        current_FT = 0.0
        return_latency = []
        for item in plan:
            i = item[0]
            task = task_graph.task_list[i]
            x = item[1]
            if x == 0:
                if len(task_graph.pre_task_sets[i]) != 0:
                    start_time = max(local_available_time,
                                     max([max(FT_locally[j], FT_wr[j]) for j in task_graph.pre_task_sets[i]]))
                else:
                    start_time = local_available_time
                T_l[i] = self.resource_cluster.locally_execution_cost(task.processing_data_size)
                FT_locally[i] = start_time + T_l[i]
                local_available_time = FT_locally[i]
                task_finish_time = FT_locally[i]
            else:
                if len(task_graph.pre_task_sets[i]) != 0:
                    ws_start_time = max(ws_available_time,
                                        max([max(FT_locally[j], FT_ws[j]) for j in task_graph.pre_task_sets[i]]))
                    T_ul[i] = self.resource_cluster.up_transmission_cost(task.processing_data_size)
                    ws_finish_time = ws_start_time + T_ul[i]
                    FT_ws[i] = ws_finish_time
                    ws_available_time = ws_finish_time
                    cloud_start_time = max(cloud_available_time,
                                           max([max(FT_ws[i], FT_cloud[j]) for j in task_graph.pre_task_sets[i]]))
                    cloud_finish_time = cloud_start_time + self.resource_cluster.mec_execution_cost(
                        task.processing_data_size)
                    FT_cloud[i] = cloud_finish_time
                    cloud_available_time = cloud_finish_time
                    wr_start_time = FT_cloud[i]
                    T_dl[i] = self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
                    wr_finish_time = wr_start_time + T_dl[i]
                    FT_wr[i] = wr_finish_time
                else:
                    ws_start_time = ws_available_time
                    T_ul[i] = self.resource_cluster.up_transmission_cost(task.processing_data_size)
                    ws_finish_time = ws_start_time + T_ul[i]
                    FT_ws[i] = ws_finish_time
                    cloud_start_time = max(cloud_available_time, FT_ws[i])
                    cloud_finish_time = cloud_start_time + self.resource_cluster.mec_execution_cost(
                        task.processing_data_size)
                    FT_cloud[i] = cloud_finish_time
                    cloud_available_time = cloud_finish_time
                    wr_start_time = FT_cloud[i]
                    T_dl[i] = self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
                    wr_finish_time = wr_start_time + T_dl[i]
                    FT_wr[i] = wr_finish_time
                task_finish_time = wr_finish_time
            delta_make_span = max(task_finish_time, current_FT) - current_FT
            current_FT = max(task_finish_time, current_FT)
            return_latency.append(delta_make_span)
        return return_latency, current_FT

    def score_func(self, cost, max_time, min_time):
        return -(cost - min_time) / (max_time - min_time)

    def get_reward_batch_step_by_step(self, action_sequence_batch, task_graph_batch,
                                      max_running_time_batch, min_running_time_batch):
        logging.debug('Start OffloadingEnvironment step')
        target_batch = []
        task_finish_time_batch = []
        for i in range(len(action_sequence_batch)):
            max_running_time = max_running_time_batch[i]
            min_running_time = min_running_time_batch[i]
            task_graph = task_graph_batch[i]
            self.resource_cluster.reset()
            plan = action_sequence_batch[i]
            cost, task_finish_time = self.get_scheduling_cost_step_by_step(plan, task_graph)
            latency = self.score_func(cost, max_running_time, min_running_time)
            score = np.array(latency)
            target_batch.append(score)
            task_finish_time_batch.append(task_finish_time)

        target_batch = np.array(target_batch)
        return target_batch, task_finish_time_batch

    def greedy_solution(self):
        result_plan = []
        finish_time_batches = []
        for task_graph_batch in self.task_graphs_batches:
            plan_batches = []
            finish_time_plan = []
            for task_graph in task_graph_batch:
                cloud_available_time = 0.0
                ws_available_time = 0.0
                local_available_time = 0.0
                ft_cloud = [0] * task_graph.task_number
                ft_ws = [0] * task_graph.task_number
                ft_locally = [0] * task_graph.task_number
                ft_wr = [0] * task_graph.task_number
                plan = []
                for i in task_graph.prioritize_sequence:
                    task = task_graph.task_list[i]
                    if len(task_graph.pre_task_sets[i]) != 0:
                        start_time = max(local_available_time,
                                         max([max(ft_locally[j], ft_wr[j]) for j in task_graph.pre_task_sets[i]]))
                    else:
                        start_time = local_available_time

                    local_running_time = self.resource_cluster.locally_execution_cost(task.processing_data_size)
                    ft_locally[i] = start_time + local_running_time
                    if len(task_graph.pre_task_sets[i]) != 0:
                        ws_start_time = max(ws_available_time,
                                            max([max(ft_locally[j], ft_ws[j]) for j in task_graph.pre_task_sets[i]]))
                        ft_ws[i] = ws_start_time + self.resource_cluster.up_transmission_cost(task.processing_data_size)
                        cloud_start_time = max(cloud_available_time,
                                               max([max(ft_ws[i], ft_cloud[j]) for j in task_graph.pre_task_sets[i]]))
                        cloud_finish_time = cloud_start_time + self.resource_cluster.mec_execution_cost(
                            task.processing_data_size)
                        ft_cloud[i] = cloud_finish_time
                        wr_start_time = ft_cloud[i]
                        wr_finish_time = wr_start_time + self.resource_cluster. \
                            dl_transmission_cost(task.transmission_data_size)
                        ft_wr[i] = wr_finish_time
                    else:
                        ws_start_time = ws_available_time
                        ws_finish_time = ws_start_time + self.resource_cluster. \
                            up_transmission_cost(task.processing_data_size)
                        ft_ws[i] = ws_finish_time

                        cloud_start_time = max(cloud_available_time, ft_ws[i])
                        ft_cloud[i] = cloud_start_time + self.resource_cluster.mec_execution_cost(
                            task.processing_data_size)
                        ft_wr[i] = ft_cloud[i] + self.resource_cluster.dl_transmission_cost(task.transmission_data_size)

                    if ft_locally[i] < ft_wr[i]:
                        action = 0
                        local_available_time = ft_locally[i]
                        ft_wr[i] = 0.0
                        ft_cloud[i] = 0.0
                        ft_ws[i] = 0.0
                    else:
                        action = 1
                        ft_locally[i] = 0.0
                        cloud_available_time = ft_cloud[i]
                        ws_available_time = ft_ws[i]
                    plan.append((i, action))

                finish_time = max(max(ft_wr), max(ft_locally))
                plan_batches.append(plan)
                finish_time_plan.append(finish_time)

            finish_time_batches.append(finish_time_plan)
            result_plan.append(plan_batches)

        return result_plan, finish_time_batches
    def get_running_cost(self, action_sequence_batch, task_graph_batch):
        logging.debug('Start get_running_cost')
        cost_batch = []
        for action_sequence, task_graph in zip(action_sequence_batch,
                                               task_graph_batch):
            plan_sequence = []

            for action, task_id in zip(action_sequence,
                                       task_graph.prioritize_sequence):
                plan_sequence.append((task_id, action))

                _, task_finish_time = self.get_scheduling_cost_step_by_step(plan_sequence, task_graph)

            cost_batch.append(task_finish_time)
        return cost_batch

    def get_all_locally_execute_time(self):
        logging.debug('Start get_all_locally_execute_time')
        running_cost = []
        for task_graph_batch, encode_batch in zip(self.task_graphs_batches, self.encoder_batches):
            batch_size = encode_batch.shape[0]
            sequence_length = encode_batch.shape[1]

            scheduling_action = np.zeros(shape=(batch_size, sequence_length), dtype=np.int32)
            running_cost_batch = self.get_running_cost(scheduling_action, task_graph_batch)
            running_cost.append(np.mean(running_cost_batch))

        return running_cost

    def get_all_mec_execute_time(self):
        logging.debug('Start get_all_mec_execute_time')
        running_cost = []

        for task_graph_batch, encode_batch in zip(self.task_graphs_batches, self.encoder_batches):
            batch_size = encode_batch.shape[0]
            sequence_length = encode_batch.shape[1]

            scheduling_action = np.ones(shape=(batch_size, sequence_length), dtype=np.int32)
            running_cost_batch = self.get_running_cost(scheduling_action, task_graph_batch)

            running_cost.append(np.mean(running_cost_batch))
        return running_cost

    def greedy_solution_for_current_task(self):
        result_plan, finish_time_batchs = self.greedy_solution()
        return result_plan[self.task_id], finish_time_batchs[self.task_id]
