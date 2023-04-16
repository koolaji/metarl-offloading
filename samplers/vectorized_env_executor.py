import numpy as np
import pickle as pickle
from multiprocessing import Process, Pipe
import copy
import logging

"""
This code defines a MetaIterativeEnvExecutor class that wraps multiple environments of the same kind and provides 
functionality to reset/step the environments in a vectorized manner. Internally, the environments are 
executed iteratively.

The class takes as input a meta environment object, the number of meta tasks, the number of environments 
per meta task, and the maximum length of sampled environment paths. The class initializes an array of environments, 
each of which is a deep copy of the input environment object.

The class implements the step() method, which takes a list of actions and steps the wrapped environments with 
the provided actions. The method returns a tuple of lists containing the observations, rewards, dones, and environment 
information for each environment. If the maximum path length is reached or the environment is done, the method resets 
the environment.

The class also implements the set_tasks() method, which sets a list of tasks to each environment. The method takes as 
input a list of tasks, where each task is a tuple of goal parameters.

The class also implements the reset() method, which resets the environments and returns a list of 
the new initial observations.

Overall, this code provides a useful tool for vectorizing the execution of multiple environments in 
meta-reinforcement learning.
"""


class MetaIterativeEnvExecutor(object):
    """
    Wraps multiple environments of the same kind and provides functionality to reset / step the environments
    in a vectorized manner. Internally, the environments are executed iteratively.

    Args:
        env (meta_policy_search.envs.base.MetaEnv): meta environment object
        meta_batch_size (int): number of meta tasks
        envs_per_task (int): number of environments per meta task
        max_path_length (int): maximum length of sampled environment paths - if the max_path_length is reached,
                               the respective environment is reset
    """

    def __init__(self, env, meta_batch_size, envs_per_task, max_path_length):
        self.envs = np.asarray([copy.deepcopy(env) for _ in range(meta_batch_size * envs_per_task)])
        self.ts = np.zeros(len(self.envs), dtype='int')  # time steps
        self.max_path_length = max_path_length

    def step(self, actions):
        """
        Steps the wrapped environments with the provided actions

        Args:
            actions (list): lists of actions, of length meta_batch_size x envs_per_task

        Returns
            (tuple): a length 4 tuple of lists, containing obs (np.array), rewards (float), dones (bool),
             env_infos (dict). Each list is of length meta_batch_size x envs_per_task
             (assumes that every task has same number of envs)
        The step method of the VecEnvExecutor class is used to step all the environments in the vectorized environment
        with the provided actions and return the resulting observations, rewards, dones, and environment infos.
        The method takes a list of actions as input, where each action corresponds to an environment in the vectorized
        environment. It then steps each environment in the vectorized environment with its corresponding action using
        a list comprehension and stores the resulting observations, rewards, dones, and environment infos in a list of
        tuples called all_results.
        The obs, rewards, dones, and env_infos lists are then created by unpacking the all_results list of tuples using
        the map and zip functions. The resulting lists contain the observations, rewards, dones, and environment infos
        for each environment in the vectorized environment.
        The method then checks if any of the environments have reached their maximum path length or are done.
        If an environment has reached its maximum path length or is done, it resets the environment and sets its time
        step counter to zero. This ensures that each environment in the vectorized environment is reset when it reaches
        the end of a trajectory or when it is done.
        Finally, the method returns the resulting observations, rewards, dones, and environment infos as a tuple of
        lists.
        """
        assert len(actions) == self.num_envs

        all_results = [env.step(a) for (a, env) in zip(actions, self.envs)]

        # stack results split to obs, rewards, ...
        obs, rewards, dones, env_infos = list(map(list, zip(*all_results)))

        # reset env when done or max_path_length reached
        dones = np.asarray(dones)
        self.ts += 1
        dones = np.logical_or(self.ts >= self.max_path_length, dones)
        for i in np.argwhere(dones).flatten():
            obs[i] = self.envs[i].reset()
            self.ts[i] = 0

        return obs, rewards, dones, env_infos

    def set_tasks(self, tasks):
        """
        Sets a list of tasks to each environment

        Args:
            tasks (list): list of the tasks for each environment
        This code is used to set the task for each environment created in the previous line of code.
        It first splits the numpy array of environments (self.envs) into a list of arrays, with each array
        containing the environments for a particular task. The number of arrays in the list is equal to the number of
        tasks. This is done using the numpy.split function.

        For each task and the corresponding array of environments, the code then sets the task for each
        environment using the set_task method of the environment object. The set_task method is typically
        defined by the user and is specific to the environment being used. It sets the task-specific parameters of
        the environment, such as the goal state or the reward function, based on the task passed as an argument.

        Overall, this code sets the task for each environment created in the previous line of code, which allows for
        task-specific training and evaluation of the meta-learned policy.
        """
        # logging.debug("set_tasks")
        envs_per_task = np.split(self.envs, len(tasks))
        # for env_per_task in envs_per_task:
        #     logging.debug("set_tasks envs_per_task %s", env_per_task)
        for task, envs in zip(tasks, envs_per_task):
            for env in envs:
                env.set_task(task)
                # logging.debug("set_tasks task envs task_id %s %s %s %s", task, env, env.task_id, env.graph_file_paths)

    def reset(self):
        """
        Resets the environments

        Returns:
            (list): list of (np.ndarray) with the new initial observations.
        The reset method of the VecEnvExecutor class is used to reset all the environments in the vectorized environment
        and return their initial observations.
        In this implementation, the method first calls the reset method of each individual environment in the vectorized
        environment using a list comprehension. This returns a list of numpy arrays containing the initial observations
        for each environment.
        Then, the ts attribute of the VecEnvExecutor object is set to zero. This attribute is a numpy array that keeps
        track of the current time step of each environment in the vectorized environment.
        Finally, the list of initial observations is returned.
        """
        obses = [env.reset() for env in self.envs]
        self.ts[:] = 0
        logging.debug("obses %s", len(obses))
        return obses

    @property
    def num_envs(self):
        """
        Number of environments

        Returns:
            (int): number of environments
        """
        return len(self.envs)


class MetaParallelEnvExecutor(object):
    """
    Wraps multiple environments of the same kind and provides functionality to reset / step the environments
    in a vectorized manner. Thereby the environments are distributed among meta_batch_size processes and
    executed in parallel.

    Args:
        env (meta_policy_search.envs.base.MetaEnv): meta environment object
        meta_batch_size (int): number of meta tasks
        envs_per_task (int): number of environments per meta task
        max_path_length (int): maximum length of sampled environment paths - if the max_path_length is reached,
                             the respective environment is reset
    """

    def __init__(self, env, meta_batch_size, envs_per_task, max_path_length):
        self.n_envs = meta_batch_size * envs_per_task
        self.meta_batch_size = meta_batch_size
        self.envs_per_task = envs_per_task
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(meta_batch_size)])
        seeds = np.random.choice(range(10**6), size=meta_batch_size, replace=False)

        self.ps = [
            Process(target=worker, args=(work_remote, remote, pickle.dumps(env), envs_per_task, max_path_length, seed))
            for (work_remote, remote, seed) in zip(self.work_remotes, self.remotes, seeds)]  # Why pass work remotes?

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions):
        """
        Executes actions on each env

        Args:
            actions (list): lists of actions, of length meta_batch_size x envs_per_task

        Returns
            (tuple): a length 4 tuple of lists, containing obs (np.array), rewards (float), dones (bool), env_infos (dict)
                      each list is of length meta_batch_size x envs_per_task (assumes that every task has same number of envs)
        """
        assert len(actions) == self.num_envs

        # split list of actions in list of list of actions per meta tasks
        chunks = lambda l, n: [l[x: x + n] for x in range(0, len(l), n)]
        actions_per_meta_task = chunks(actions, self.envs_per_task)

        # step remote environments
        for remote, action_list in zip(self.remotes, actions_per_meta_task):
            remote.send(('step', action_list))

        results = [remote.recv() for remote in self.remotes]

        obs, rewards, dones, env_infos = map(lambda x: sum(x, []), zip(*results))

        return obs, rewards, dones, env_infos

    def reset(self):
        """
        Resets the environments of each worker

        Returns:
            (list): list of (np.ndarray) with the new initial observations.
        """
        for remote in self.remotes:
            remote.send(('reset', None))
        return sum([remote.recv() for remote in self.remotes], [])

    def set_tasks(self, tasks=None):
        """
        Sets a list of tasks to each worker

        Args:
            tasks (list): list of the tasks for each worker
        """
        for remote, task in zip(self.remotes, tasks):
            remote.send(('set_task', task))
        for remote in self.remotes:
            remote.recv()

    @property
    def num_envs(self):
        """
        Number of environments

        Returns:
            (int): number of environments
        """
        return self.n_envs


def worker(remote, parent_remote, env_pickle, n_envs, max_path_length, seed):
    """
    Instantiation of a parallel worker for collecting samples. It loops continually checking the task that the remote
    sends to it.

    Args:
        remote (multiprocessing.Connection):
        parent_remote (multiprocessing.Connection):
        env_pickle (pkl): pickled environment
        n_envs (int): number of environments per worker
        max_path_length (int): maximum path length of the task
        seed (int): random seed for the worker
    """
    parent_remote.close()

    envs = [pickle.loads(env_pickle) for _ in range(n_envs)]
    np.random.seed(seed)

    ts = np.zeros(n_envs, dtype='int')

    while True:
        # receive command and data from the remote
        cmd, data = remote.recv()

        # do a step in each of the environment of the worker
        if cmd == 'step':
            all_results = [env.step(a) for (a, env) in zip(data, envs)]
            obs, rewards, dones, infos = map(list, zip(*all_results))
            ts += 1
            for i in range(n_envs):
                if dones[i] or (ts[i] >= max_path_length):
                    dones[i] = True
                    obs[i] = envs[i].reset()
                    ts[i] = 0
            remote.send((obs, rewards, dones, infos))

        # reset all the environments of the worker
        elif cmd == 'reset':
            obs = [env.reset() for env in envs]
            ts[:] = 0
            remote.send(obs)

        # set the specified task for each of the environments of the worker
        elif cmd == 'set_task':
            for env in envs:
                env.set_task(data)
            remote.send(None)

        # close the remote and stop the worker
        elif cmd == 'close':
            remote.close()
            break

        else:
            raise NotImplementedError
