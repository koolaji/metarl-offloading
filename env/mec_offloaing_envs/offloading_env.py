from env.base import MetaEnv
from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph
import numpy as np
import logging


def _computation_cost(data, processing_power):
    computation_time = data / processing_power

    return computation_time


class Resources(object):
    """
    The Resources class provides a simple implementation of a Mobile Edge 
    Computing (MEC) system, where the processing of data can be offloaded 
    from mobile devices to the MEC server, depending on the available 
    resources and network conditions. The class models the computation 
    capacities of the MEC server and mobile device, as well as the 
    wireless uplink and downlink bandwidths.
    The methods up_transmission_cost, dl_transmission_cost, 
    locally_execution_cost, and mec_execution_cost calculate the 
    transmission time for data sent between the mobile device and 
    MEC server, as well as the time to execute data on the device or server. 
    These methods rely on the _computation_cost function, which is not 
    defined in this class.
    Overall, the Resources class provides a good starting point for 
    modeling an MEC system, but it may require further customization and 
    implementation of additional methods to meet specific use cases.
    """

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


class OffloadingEnvironment(MetaEnv):
    def __init__(self, resource_cluster, batch_size,
                 graph_number,
                 graph_file_paths, time_major):
        self.resource_cluster = resource_cluster
        self.task_graphs_batches = []
        self.encoder_batches = []
        self.encoder_lengths = []
        self.decoder_full_lengths = []
        self.max_running_time_batches = []
        self.min_running_time_batches = []
        self.graph_file_paths = graph_file_paths

        # load all the task graphs into the environment
        for graph_file_path in graph_file_paths:
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

        # set the file paht of task graphs.
        self.graph_file_paths = graph_file_paths
        self.graph_number = graph_number

        self.local_exe_time = self.get_all_locally_execute_time()
        self.mec_exe_time = self.get_all_mec_execute_time()

    def sample_tasks(self, n_tasks):
        """
        Samples task of the meta-environment

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        This line of code generates a random subset of tasks from the total number of tasks available.
        It uses the numpy.random.choice function to randomly select n_tasks number of tasks from
        np.arrange(self.total_task), which is an array of integers from 0 to self.total_task-1. The replacement=False
        argument ensures that the same task is not selected more than once.

        For example, if self.total_task = 10 and n_tasks = 3, this line of code could generate a random subset of
        tasks [2, 5, 8].
        """
        return np.random.choice(np.arange(self.total_task), n_tasks, replace=False)

    # def merge_graphs(self):
    #     encoder_batches = []
    #     encoder_lengths = []
    #     task_graphs_batches = []
    #     decoder_full_lengths =[]
    #     max_running_time_batches = []
    #     min_running_time_batches = []
    #
    #     for encoder_batch, encoder_length, task_graphs_batch, \
    #         decoder_full_length, max_running_time_batch, \
    #         min_running_time_batch in zip(self.encoder_batches, self.encoder_lengths,
    #                                       self.task_graphs_batches, self.decoder_full_lengths,
    #                                       self.max_running_time_batches, self.min_running_time_batches):
    #         encoder_batches += encoder_batch.tolist()
    #         encoder_lengths += encoder_length.tolist()
    #         task_graphs_batches += task_graphs_batch
    #         decoder_full_lengths += decoder_full_length.tolist()
    #         max_running_time_batches += max_running_time_batch
    #         min_running_time_batches += min_running_time_batch
    #
    #     self.encoder_batches = np.array([encoder_batches])
    #     self.encoder_lengths = np.array([encoder_lengths])
    #     self.task_graphs_batches = [task_graphs_batches]
    #     self.decoder_full_lengths = np.array([decoder_full_lengths])
    #     self.max_running_time_batches = np.array([max_running_time_batches])
    #     self.min_running_time_batches = np.array([min_running_time_batches])

    def set_task(self, task):
        """
        Sets the specified task to the current environment

        Args:
            task: task of the meta-learning environment
        """
        self.task_id = task

    def get_task(self):
        """
        Gets the task that the agent is performing in the current environment

        Returns:
            task: task of the meta-learning environment
        """
        return self.graph_file_paths[self.task_id]

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
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
        """Resets the state of the environment and returns an initial observation.

        Returns:
            observation (object): the initial observation.
        """
        # reset the resource environment.
        self.resource_cluster.reset()

        return np.array(self.encoder_batches[self.task_id])

    def render(self, mode='human'):
        pass

    def generate_point_batch_for_random_graphs(self, batch_size, graph_number, graph_file_path, time_major):
        """
        This method generates batches of encoded task graphs with associated 
        metadata for a set of randomly generated task graphs.
        The method takes several arguments:
        batch_size: The number of task graphs per batch.
        graph_number: The total number of task graphs to generate.
        graph_file_path: The file path to the Graphviz files that contain the task graphs.
        time_major: A boolean indicating whether the data in the batch should be 
        time-major (True) or batch-major (False).
        The method first initializes several lists and arrays to store the encoded task graphs,
        the task graph objects themselves, and metadata such as the maximum and 
        minimum running times for each task graph.
        The method then iterates over each task graph, creates an OffloadingTaskGraph 
        object for the graph, and calculates the maximum and minimum running times 
        for the graph. It then generates an encoding of the task graph using the 
        encode_point_sequence_with_ranking_and_cost method of the OffloadingTaskGraph 
        object and appends it to a list of encoded task graphs. It also appends 
        the task graph object to a list of task graphs.
        Next, the method groups the encoded task graphs and task graph objects into 
        batches of size batch_size. For each batch, it creates an array of encoded 
        task graphs, and if time_major is True, it transposes the array to make it time-major. 
        It also calculates the sequence length for each task graph in the batch and 
        appends it to a list of sequence lengths.
        Finally, the method returns several arrays containing the encoded task graph 
        batches, their sequence lengths, the corresponding task graph object batches, 
        the sequence lengths of the full decoder inputs, and the maximum and minimum 
        running times for each task graph batch.
        Overall, this method provides a way to generate batches of encoded task graphs 
        and associated metadata for use in training and evaluating machine learning models 
        in an MEC system. The implementation is clear and follows best practices, such as using 
        descriptive variable names and commenting code appropriately.
        """
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

            # the scheduling sequence will also store in self.'prioritize_sequence'
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

        return encoder_batches, encoder_lengths, task_graph_batches, \
               decoder_full_lengths, max_running_time_batches, \
               min_running_time_batches

    def calculate_max_min_runningcost(self, max_data_size, min_data_size):
        max_time = max([self.resource_cluster.up_transmission_cost(max_data_size),
                        self.resource_cluster.dl_transmission_cost(max_data_size),
                        self.resource_cluster.locally_execution_cost(max_data_size)])

        min_time = self.resource_cluster.mec_execution_cost(min_data_size)

        return max_time, min_time

    def get_scheduling_cost_step_by_step(self, plan, task_graph):
        """
        This code defines a method called get_scheduling_cost_step_by_step that calculates the latency 
        of executing a set of tasks on a distributed system. The method takes two arguments: 
        plan, which is a list of tuples representing the scheduling plan, and task_graph, 
        which is a directed acyclic graph representing the dependencies between the tasks. 
        The method returns a list of latencies for each task in the plan, as well as the total time 
        it takes to execute all the tasks.
        The method first initializes variables for available time on the cloud, the wireless network, 
        and the local processor. It then initializes variables for the running time on the local processor, 
        sending channel, and receiving channel for each task, as well as the finish time on the cloud, 
        sending channel, local processor, and receiving channel for each task.
        The method then iterates through the scheduling plan and for each task, it checks whether 
        the task is scheduled to run locally or on the cloud. If the task is scheduled to run locally, 
        it calculates the start time for the task based on the available time on the local processor and 
        the finish times of its predecessors, then calculates the running time on the local processor and 
        the finish time on the local processor for the task. If the task is scheduled to run on the cloud, 
        it calculates the start time for the task on the sending channel based on the available time on 
        the wireless network and the finish times of its predecessors, then calculates the running time on 
        the sending channel, finish time on the sending channel, start time on the cloud, finish time on 
        the cloud, start time on the receiving channel, and finish time on the receiving channel for the task.
        The method then calculates the delta make-span for the task, which is the time it takes to complete 
        the current task after completing all its predecessors. It also updates the current finish time, 
        which is the time it takes to complete all the tasks up to the current task. Finally, the method 
        returns the list of latencies for each task and the total time it takes to execute all the tasks.
        """
        cloud_available_time = 0.0
        ws_available_time = 0.0
        local_available_time = 0.0

        # running time on local processor
        T_l = [0] * task_graph.task_number
        # running time on sending channel
        T_ul = [0] * task_graph.task_number
        # running time on receiving channel
        T_dl = [0] * task_graph.task_number

        # finish time on cloud for each task
        FT_cloud = [0] * task_graph.task_number
        # finish time on sending channel for each task
        FT_ws = [0] * task_graph.task_number
        # finish time locally for each task
        FT_locally = [0] * task_graph.task_number
        # finish time receiving channel for each task
        FT_wr = [0] * task_graph.task_number
        current_FT = 0.0
        # total_energy = 0.0
        return_latency = []
        # return_energy = []

        for item in plan:
            i = item[0]
            task = task_graph.task_list[i]
            x = item[1]

            # locally scheduling
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

                # calculate the energy consumption
                # energy_consumption = T_l[i] * self.rho * (self.f_l ** self.zeta)
            # mcc scheduling
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
                    # print("task {}, Cloud finish time {}".format(i, FT_cloud[i]))
                    cloud_available_time = cloud_finish_time

                    wr_start_time = FT_cloud[i]
                    T_dl[i] = self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
                    wr_finish_time = wr_start_time + T_dl[i]
                    FT_wr[i] = wr_finish_time

                    # calculate the energy consumption
                    # energy_consumption = T_ul[i] * self.ptx + T_dl[i] * self.prx

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

                    # calculate the energy consumption
                    # energy_consumption = T_ul[i] * self.ptx + T_dl[i] * self.prx

                task_finish_time = wr_finish_time

            # print("task  {} finish time is {}".format(i , task_finish_time))
            delta_make_span = max(task_finish_time, current_FT) - current_FT
            current_FT = max(task_finish_time, current_FT)
            return_latency.append(delta_make_span)

        return return_latency, current_FT

    def score_func(self, cost, max_time, min_time):
        return -(cost - min_time) / (max_time - min_time)

    def get_reward_batch_step_by_step(self, action_sequence_batch, task_graph_batch,
                                      max_running_time_batch, min_running_time_batch):
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
            # print("score is", score)
            target_batch.append(score)
            task_finish_time_batch.append(task_finish_time)

        target_batch = np.array(target_batch)
        return target_batch, task_finish_time_batch

    def greedy_solution(self):
        """
        This class appears to represent a scheduler in an MEC system that generates a plan for 
        executing a set of task graphs on available resources. The generate_schedule method of 
        this class uses a heuristic algorithm to generate a plan for executing each task 
        graph in the set.
        The method first initializes an empty list result_plan and finish_time_batches. It then 
        iterates over each batch of task graphs in self.task_graphs_batches. For each batch, 
        it initializes an empty list plan_batches and finish_time_plan.
        The method then iterates over each task graph in the batch and initializes variables 
        to track available execution times for each resource type (local, MEC, cloud), as well as 
        lists to store the finish times for each task on each resource type. It also initializes 
        an empty plan list to store the execution plan for the task graph.
        For each task in the task graph, the method calculates the finish time for executing 
        the task on each resource type (locally, on an MEC server, or on the cloud) based on 
        the finish times of its predecessors. It then chooses the execution option (local or remote) 
        that results in the earliest finish time for the task. The method adds the task and its 
        chosen execution option to the plan list and updates the available execution times for 
        each resource type.
        After iterating over all tasks in the task graph, the method calculates the finish time 
        for the entire task graph and appends the plan list and finish time to plan_batches and 
        finish_time_plan, respectively.
        Finally, after iterating over all task graphs in the batch, the method appends 
        finish_time_plan to finish_time_batches and plan_batches to result_plan.
        Overall, this class provides a way to generate an execution plan for a set of task graphs 
        in an MEC system using a heuristic algorithm. 
        """
        result_plan = []
        finish_time_batches = []
        for task_graph_batch in self.task_graphs_batches:
            plan_batches = []
            finish_time_plan = []
            for task_graph in task_graph_batch:
                cloud_available_time = 0.0
                ws_available_time = 0.0
                local_available_time = 0.0

                # finish time on cloud for each task
                ft_cloud = [0] * task_graph.task_number
                # finish time on sending channel for each task
                ft_ws = [0] * task_graph.task_number
                # finish time locally for each task
                ft_locally = [0] * task_graph.task_number
                # finish time receiving channel for each task
                ft_wr = [0] * task_graph.task_number
                plan = []

                for i in task_graph.prioritize_sequence:
                    task = task_graph.task_list[i]

                    # calculate the local finish time
                    if len(task_graph.pre_task_sets[i]) != 0:
                        start_time = max(local_available_time,
                                         max([max(ft_locally[j], ft_wr[j]) for j in task_graph.pre_task_sets[i]]))
                    else:
                        start_time = local_available_time

                    local_running_time = self.resource_cluster.locally_execution_cost(task.processing_data_size)
                    ft_locally[i] = start_time + local_running_time

                    # calculate the remote finish time
                    if len(task_graph.pre_task_sets[i]) != 0:
                        ws_start_time = max(ws_available_time,
                                            max([max(ft_locally[j], ft_ws[j]) for j in task_graph.pre_task_sets[i]]))
                        ft_ws[i] = ws_start_time + self.resource_cluster.up_transmission_cost(task.processing_data_size)
                        cloud_start_time = max(cloud_available_time,
                                               max([max(ft_ws[i], ft_cloud[j]) for j in task_graph.pre_task_sets[i]]))
                        cloud_finish_time = cloud_start_time + self.resource_cluster.mec_execution_cost(
                            task.processing_data_size)
                        ft_cloud[i] = cloud_finish_time
                        # print("task {}, Cloud finish time {}".format(i, ft_cloud[i]))
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

    def calculate_optimal_solution(self):
        # Finding the optimal solution via exhausting search the solution space.
        def exhaustion_plans(n):
            plan_batch = []

            for i in range(2 ** n):
                plan_str = bin(i)
                plan = []

                for x in plan_str[2:]:
                    plan.append(int(x))

                while len(plan) < n:
                    plan.insert(0, 0)
                plan_batch.append(plan)
            return plan_batch

        n = self.task_graphs_batches[0][0].task_number
        plan_batch = exhaustion_plans(n)

        print("exhausted plan size: ", len(plan_batch))

        task_graph_optimal_costs = []
        optimal_plan = []

        for task_graph_batch in self.task_graphs_batches:
            task_graph_batch_cost = []
            for task_graph in task_graph_batch:
                plans_costs = []
                prioritize_plan = []

                for plan in plan_batch:
                    plan_sequence = []
                    for action, task_id in zip(plan, task_graph.prioritize_sequence):
                        plan_sequence.append((task_id, action))

                    cos, task_finish_time = self.get_scheduling_cost_step_by_step(plan_sequence, task_graph)
                    plans_costs.append(task_finish_time)

                    prioritize_plan.append(plan_sequence)

                graph_min_cost = min(plans_costs)

                optimal_plan.append(prioritize_plan[np.argmin(plans_costs)])

                task_graph_batch_cost.append(graph_min_cost)

            print("task_graph_batch cost shape is {}".format(np.array(task_graph_batch_cost).shape))
            avg_minimal_cost = np.mean(task_graph_batch_cost)

            task_graph_optimal_costs.append(avg_minimal_cost)

        self.optimal_solution = task_graph_optimal_costs
        return task_graph_optimal_costs, optimal_plan

    def get_running_cost(self, action_sequence_batch, task_graph_batch):
        """
        This is a method definition in Python. It defines a method called get_running_cost that 
        takes two arguments: action_sequence_batch and task_graph_batch. It returns a list of 
        floating-point numbers representing the execution time for each task graph in the batch.
        The method iterates over pairs of action_sequence and task_graph in the action_sequence_batch 
        and task_graph_batch, respectively. It assumes that action_sequence_batch and task_graph_batch 
        are arrays or tensors containing batched action sequences and task graphs, respectively.
        For each pair of action_sequence and task_graph, the method creates an empty list called 
        plan_sequence, which will be used to store a sequence of (task_id, action) tuples. 
        It then iterates over pairs of action and task_id in action_sequence and 
        task_graph.prioritize_sequence, respectively. For each pair, it appends a tuple containing 
        the task ID and action to plan_sequence. It then calls the get_scheduling_cost_step_by_step 
        method with plan_sequence and task_graph to compute the finish time for the task sequence.
        The finish time is appended to a list called cost_batch, which will store the finish times 
        for all task graphs in the batch. Finally, the method returns cost_batch, which is a list of 
        floating-point numbers representing the finish times for each task graph in the batch.
        """
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
        running_cost = []
        for task_graph_batch, encode_batch in zip(self.task_graphs_batches, self.encoder_batches):
            batch_size = encode_batch.shape[0]
            sequence_length = encode_batch.shape[1]

            scheduling_action = np.zeros(shape=(batch_size, sequence_length), dtype=np.int32)
            running_cost_batch = self.get_running_cost(scheduling_action, task_graph_batch)
            running_cost.append(np.mean(running_cost_batch))

        return running_cost

    def get_all_mec_execute_time(self):
        """
        This is a method definition in Python. It defines a method called get_all_mec_execute_time 
        that takes no arguments and returns a list of floating-point numbers representing 
        the average execution time for each batch of tasks.
        The method iterates over pairs of task_graph_batch and encode_batch, which are assumed 
        to be arrays or tensors containing batched task graphs and their corresponding encoded 
        representations, respectively. For each pair, it computes the batch size and sequence 
        length of the encoded representations, creates a scheduling action array of ones with 
        the same shape as the encoded representations, and calls the get_running_cost method 
        with the scheduling action and task graph batch to compute the running cost for the batch. 
        The running cost is then appended to a list of running costs for all batches, and the method 
        returns the list of average running costs for each batch.
        """
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
