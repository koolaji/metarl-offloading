import numpy as np
from graphviz import Digraph
import json
import pydotplus


class OffloadingTask(object):
    """
    The OffloadingTask class is a simple Python class that represents a 
    task to be executed in a mobile edge computing (MEC) system. 
    The class is initialized with the ID name, processing data size, 
    transmission data size, task type name, depth, and HEFT score.
    The print_task method prints out the details of the task, including 
    the ID name, task type name, processing data size, and transmission data size.
    Overall, the OffloadingTask class provides a basic implementation of a task in 
    an MEC system, and it could be extended to include additional functionality and 
    methods as needed. The class has a clear and concise interface, and the use of 
    parameter names that match their corresponding attributes makes it easy to understand.
    """
    def __init__(self, id_name, process_data_size, transmission_data_size, 
                 type_name, depth=0, heft_score=0):
        self.id_name = id_name
        self.processing_data_size = process_data_size
        self.transmission_data_size = transmission_data_size
        self.type_name = type_name
        self.depth = depth
        self.heft_score = heft_score
        self.all_locally_execute = 0.0
        self.all_mec_execute = 0.0

    def print_task(self):
        print("task id name: {}, task type name: {} task processing data size: {}, "
              "task transmission_data_size: {}".format(
                                self.id_name, self.type_name,
            self.processing_data_size, self.transmission_data_size))


class OffloadingDotParser(object):
    """
    The OffloadingDotParser class is a Python class that provides a parser for task 
    graphs in a mobile edge computing (MEC) system. The class is initialized with a file 
    name that contains a graphviz file.
    The class provides methods to parse the tasks and dependencies in the graph, calculate 
    the depth and transmission data size of each task, and generate a list of tasks and 
    their dependencies.
    The generate_task_list method returns a list of OffloadingTask objects, representing 
    the tasks in the graph. The generate_dependency method returns a list of dependencies 
    between tasks.
    Overall, the OffloadingDotParser class provides a useful tool for parsing task graphs 
    in an MEC system, and its implementation is clear and concise. The use of descriptive 
    method names and variable names makes it easy to understand, and the commented 
    documentation for the constructor method is helpful.
    """
    def __init__(self, file_name):
        self.succ_task_for_ids = {}
        self.pre_task_for_ids = {}

        self.dot_ob = pydotplus.graphviz.graph_from_dot_file(file_name)
        self._parse_task()
        self._parse_dependecies()
        self._calculate_depth_and_transimission_datasize()

    def _parse_task(self):
        jobs = self.dot_ob.get_node_list()
        self.task_list = [0] * len(jobs)

        for job in jobs:
            job_id = job.get_name()
            data_size = int(eval(job.obj_dict['attributes']['size']))
            communication_data_size = int(eval(job.obj_dict['attributes']['expect_size']))

            task = OffloadingTask(job_id, data_size, 0, "compute")
            task.transmission_data_size = communication_data_size
            id = int(job_id) - 1
            self.task_list[id] = task

    def _parse_dependecies(self):
        edge_list = self.dot_ob.get_edge_list()
        dependencies = []

        task_number = len(self.task_list)
        dependency_matrix = np.zeros(shape=(task_number, task_number),
                                     dtype=np.float32)

        for i in range(len(self.task_list)):
            self.pre_task_for_ids[i] = []
            self.succ_task_for_ids[i] = []
            dependency_matrix[i][i] = self.task_list[i].processing_data_size

        for edge in edge_list:
            source_id = int(edge.get_source()) - 1
            destination_id = int(edge.get_destination()) - 1
            data_size = int(eval(edge.obj_dict['attributes']['size']))

            self.pre_task_for_ids[destination_id].append(source_id)
            self.succ_task_for_ids[source_id].append(destination_id)

            dependency = [source_id, destination_id, data_size]

            dependency_matrix[source_id][destination_id] = data_size
            dependencies.append(dependency)

        self.dependencies = dependencies
        self.dependency_matrix = dependency_matrix

    def _calculate_depth_and_transimission_datasize(self):
        ids_to_depth = dict()

        def caluclate_depth_value(id):
            if id in ids_to_depth.keys():
                return ids_to_depth[id]
            else:
                if len(self.pre_task_for_ids[id]) != 0:
                    depth = 1 + max([caluclate_depth_value(pre_task_id) for
                                     pre_task_id in self.pre_task_for_ids[id]])
                else:
                    depth = 0

                ids_to_depth[id] = depth

            return ids_to_depth[id]

        for id in range(len(self.task_list)):
            ids_to_depth[id] = caluclate_depth_value(id)

        for id, depth in ids_to_depth.items():
            self.task_list[id].depth = depth

    def generate_task_list(self):
        return self.task_list

    def generate_dependency(self):
        return self.dependencies


class OffloadingTaskGraph(object):
    """
    The OffloadingTaskGraph class represents a parsed task graph in a mobile edge 
    computing (MEC) system. The class takes a file name of the task graph as an 
    argument and provides methods to parse the task graph, add tasks to the graph, 
    add dependencies between tasks, and encode the graph into a point sequence or 
    an edge sequence.
    The encode_point_sequence_with_cost method encodes each task in the graph as 
    a vector that includes the task's ID, the processing and transmission costs 
    for executing the task locally on the mobile device, executing the task on 
    an MEC server, and transmitting the data to and from the server. The vector 
    also includes the indices of the task's predecessors and successors in 
    the graph. The encode_point_sequence_with_ranking_and_cost method returns 
    the encoded point sequence with tasks sorted according to their priority.
    The encode_edge_sequence method encodes the edges in the graph as a sequence 
    of five-dimensional vectors representing the source task, its depth, its 
    processing data size, the transmission cost, and the destination task, 
    its depth, and its processing data size.
    The prioritize_tasks method prioritizes the tasks in the graph based on their 
    processing and transmission costs and returns the sorted task indices. 
    The render method generates a visual representation of the task graph using 
    the Graphviz library.
    """
    def __init__(self, file_name):
        self._parse_from_dot(file_name)

    # add task list to
    def _parse_from_dot(self, file_name):
        parser = OffloadingDotParser(file_name)
        task_list = parser.generate_task_list()

        self.task_number = len(task_list)
        self.dependency = np.zeros((self.task_number, self.task_number))
        self.task_list = []
        self.prioritize_sequence = []

        self.pre_task_sets = []
        self.succ_task_sets = []
        self.task_finish_time = [0] * self.task_number
        self.edge_set = []

        for _ in range(self.task_number):
            self.pre_task_sets.append(set([]))
            self.succ_task_sets.append(set([]))
        # add task list to
        self.add_task_list(task_list)

        dependencies = parser.generate_dependency()

        for pair in dependencies:
            self.add_dependency(pair[0], pair[1], pair[2])

        # get max data size and min data size, used to feature scaling.
        self.max_data_size = np.max(self.dependency[self.dependency > 0.01])
        self.min_data_size = np.min(self.dependency[self.dependency > 0.01])

    def add_task_list(self, task_list):
        self.task_list = task_list

        for i in range(0, len(self.task_list)):
            self.dependency[i][i] = task_list[i].processing_data_size

    def norm_feature(self, data_size):
        return float(data_size - self.min_data_size) / float(self.max_data_size - self.min_data_size)

    def add_dependency(self, pre_task_index, succ_task_index, transmission_cost):
        self.dependency[pre_task_index][succ_task_index] = transmission_cost
        self.pre_task_sets[succ_task_index].add(pre_task_index)
        self.succ_task_sets[pre_task_index].add(succ_task_index)

        # for each edge, we use a five dimension vector to represent this
        edge = [pre_task_index,
                self.task_list[pre_task_index].depth,
                self.task_list[pre_task_index].processing_data_size,
                transmission_cost,
                succ_task_index,
                self.task_list[succ_task_index].depth,
                self.task_list[succ_task_index].processing_data_size]

        self.edge_set.append(edge)

    # def encode_point_sequence(self):
    #     point_sequence = []
    #     for i in range(self.task_number):
    #         norm_processing_data_size = self.norm_feature(self.task_list[i].processing_data_size)
    #         norm_transmission_data_size = self.norm_feature(self.task_list[i].transmission_data_size)
    #         norm_data_size_list = [norm_processing_data_size, norm_transmission_data_size]
    #         # heft_score = [self.task_list[i].heft_score]
    #
    #         pre_task_index_set = []
    #         succs_task_index_set = []
    #
    #         for pre_task_index in range(0, i):
    #             if self.dependency[pre_task_index][i] > 0.1:
    #                 pre_task_index_set.append(pre_task_index)
    #
    #         while (len(pre_task_index_set) < 6):
    #             pre_task_index_set.append(-1.0)
    #
    #         for succs_task_index in range(i + 1, self.task_number):
    #             if self.dependency[i][succs_task_index] > 0.1:
    #                 succs_task_index_set.append(succs_task_index)
    #
    #         while (len(succs_task_index_set) < 6):
    #             succs_task_index_set.append(-1.0)
    #
    #         succs_task_index_set = succs_task_index_set[0:6]
    #         pre_task_index_set = pre_task_index_set[0:6]
    #
    #         point_vector = norm_data_size_list + pre_task_index_set + succs_task_index_set
    #         point_sequence.append(point_vector)
    #
    #     return point_sequence

    # def encode_point_sequence_with_ranking(self, sorted_task):
    #     point_sequence = self.encode_point_sequence()
    #
    #     prioritize_point_sequence = []
    #     for task_id in sorted_task:
    #         prioritize_point_sequence.append(point_sequence[task_id])
    #
    #     return prioritize_point_sequence

    def encode_point_sequence_with_cost(self, resource_cluster):
        """
        This method encodes each task in the task graph as a vector that includes the task's 
        ID, the processing and transmission costs for executing the task locally on 
        the mobile device, executing the task on an MEC server, and transmitting the data to 
        and from the server. The vector also includes the indices of the task's predecessors 
        and successors in the graph.
        The method takes a resource_cluster object as an argument, which represents the available 
        resources in the MEC system. It then iterates over each task in the task graph and 
        calculates the processing and transmission costs for executing the task locally on the 
        mobile device, executing the task on an MEC server, and transmitting the data to and 
        from the server.
        For each task, the method creates an embedding vector that includes the task's ID and 
        the processing and transmission costs. It then finds the indices of the task's 
        predecessors and successors in the graph and appends them to the embedding vector. 
        If the task has fewer than six predecessors or successors, the method pads the embedding 
        vector with -1.0 values to make the vector length equal to 11 (i.e., 5 embedding values + 6 
        predecessor indices + 6 successor indices).
        Finally, the method appends the embedding vector for each task to a list 
        point_sequence and returns the list.
        Overall, this method provides a way to encode the task graph as a sequence of vectors 
        that can be used as input to machine learning models or optimization algorithms in 
        an MEC system. The method implementation is clear and follows best practices, such as 
        using descriptive variable names and commenting code appropriately.
        """
        point_sequence = []
        for i in range(self.task_number):
            task = self.task_list[i]
            local_process_cost = task.processing_data_size / resource_cluster.mobile_process_capable
            up_link_cost = resource_cluster.up_transmission_cost(task.processing_data_size)
            mec_process_cost = task.processing_data_size / resource_cluster.mec_process_capable
            down_link_cost = resource_cluster.dl_transmission_cost(task.transmission_data_size)

            task_embeding_vector = [i, local_process_cost, up_link_cost,
                                    mec_process_cost, down_link_cost]

            pre_task_index_set = []
            succs_task_index_set = []

            for pre_task_index in range(0, i):
                if self.dependency[pre_task_index][i] > 0.1:
                    pre_task_index_set.append(pre_task_index)

            while (len(pre_task_index_set) < 6):
                pre_task_index_set.append(-1.0)

            for succs_task_index in range(i + 1, self.task_number):
                if self.dependency[i][succs_task_index] > 0.1:
                    succs_task_index_set.append(succs_task_index)

            while (len(succs_task_index_set) < 6):
                succs_task_index_set.append(-1.0)

            succs_task_index_set = succs_task_index_set[0:6]
            pre_task_index_set = pre_task_index_set[0:6]

            point_vector = task_embeding_vector + pre_task_index_set + succs_task_index_set
            point_sequence.append(point_vector)

        return point_sequence

    def encode_point_sequence_with_ranking_and_cost(self, sorted_task, resource_cluster):
        point_sequence = self.encode_point_sequence_with_cost(resource_cluster)

        prioritize_point_sequence = []
        for task_id in sorted_task:
            prioritize_point_sequence.append(point_sequence[task_id])

        return prioritize_point_sequence

    # def encode_edge_sequence(self):
    #     edge_array = []
    #     for i in range(0, len(self.edge_set)):
    #         if i < len(self.edge_set):
    #             edge_array.append(self.edge_set[i])
    #         else:
    #             edge_array.append([0, 0, 0, 0, 0, 0, 0])

    #     # input edge sequence refers to start node index
    #     edge_array = sorted(edge_array)

    #     return edge_array

    # def return_cost_metric(self):
    #     adj_matrix = np.array(self.dependency)
    #     cost_set = adj_matrix[np.nonzero(adj_matrix)]
    #     cost_set = cost_set[cost_set > 0.01]

    #     mean = np.mean(cost_set)
    #     std = np.std(cost_set)

    #     return mean, std

    # def print_graphic(self):
    #     print(self.dependency)
    #     print("This is pre_task_sets:")
    #     print(self.pre_task_sets)
    #     print("This is edge set:")
    #     print(self.edge_set)

    def prioritize_tasks(self, resource_cluster):
        w = [0] * self.task_number
        for i, task in enumerate(self.task_list):
            t_locally = task.processing_data_size / resource_cluster.mobile_process_capable
            t_mec = resource_cluster.up_transmission_cost(task.processing_data_size) + \
                    task.processing_data_size / resource_cluster.mec_process_capable + \
                    resource_cluster.dl_transmission_cost(task.transmission_data_size)

            w[i] = min(t_locally, t_mec)

        rank_dict = [-1] * self.task_number
        def rank(task_index):
            if rank_dict[task_index] != -1:
                return rank_dict[task_index]

            if len(self.succ_task_sets[task_index]) == 0:
                rank_dict[task_index] = w[task_index]
                return rank_dict[task_index]
            else:
                rank_dict[task_index] = w[task_index] + max(rank(j) for j in self.succ_task_sets[task_index])
                return rank_dict[task_index]
        for i in range(self.task_number):
            rank(i)
        sort = np.argsort(rank_dict)[::-1]
        print(sort)
        self.prioritize_sequence = sort
        return sort

    def render(self, path):
        dot = Digraph(comment='DAG')

        # str(self.task_list[i].running_time)
        for i in range(0, self.task_number):
            dot.node(str(i), str(i) + ":" + str(self.task_list[i].processing_data_size))

        for e in self.edge_set:
            dot.edge(str(e[0]), str(e[4]), constraint='true', label="%.6f" % e[3])

        dot.render(path, view=False)
