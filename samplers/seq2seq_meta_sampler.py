"""
This code defines a Seq2SeqMetaSampler class that is used for sampling trajectories for meta-reinforcement learning.
he class takes as input an environment, a policy, and various parameters such as the batch size, the maximum path
length, and the number of environments to run vectorized for each task.

The class implements the obtain_samples() method, which collects batch_size trajectories from each task by executing
the policy on the current observations and stepping the environment. The method returns a dictionary of paths, where
each path corresponds to one task and contains the observations, actions, logits, rewards, finish times, and values for
each time step in the trajectory.

The class also implements the update_tasks() method, which samples a new goal for each meta task. The method returns
a list of tasks, where each task is a tuple of goal parameters.

The sampler can be run in parallel mode or iterative mode, depending on the parallel parameter passed to the
constructor. In parallel mode, the sampler uses a MetaParallelEnvExecutor to run multiple environments in parallel for
each task. In iterative mode, the sampler uses a MetaIterativeEnvExecutor to run a single
environment iteratively for each task.

Overall, this code provides a useful tool for collecting trajectories for meta-reinforcement learning,
which is a type of reinforcement learning where the goal is to learn how to learn by adapting to new tasks quickly.
"""

from samplers.base import Sampler
from samplers.vectorized_env_executor import MetaParallelEnvExecutor, MetaIterativeEnvExecutor
# from samplers.vectorized_env_executor import  MetaIterativeEnvExecutor
from utils import utils, logger
from collections import OrderedDict

from pyprind import ProgBar
import numpy as np
import time
import itertools
import logging


class Seq2SeqMetaSampler(Sampler):
    """
    Sampler for Meta-RL

    Args:
        env (meta_policy_search.envs.base.MetaEnv) : environment object
        policy (meta_policy_search.policies.base.Policy) : policy object
        batch_size (int) : number of trajectories per task
        meta_batch_size (int) : number of meta tasks
        max_path_length (int) : max number of steps per trajectory
        envs_per_task (int) : number of envs to run vectorized for each task (influences the memory usage)
    """

    def __init__(self, env, policy, rollouts_per_meta_task, meta_batch_size, max_path_length, envs_per_task=None,
                 parallel=False):
        logging.debug('Start Seq2SeqMetaSampler')
        super(Seq2SeqMetaSampler, self).__init__(env, policy, rollouts_per_meta_task, max_path_length)
        assert hasattr(env, 'set_task')

        self.envs_per_task = rollouts_per_meta_task if envs_per_task is None else envs_per_task
        self.meta_batch_size = meta_batch_size
        self.total_samples = meta_batch_size * rollouts_per_meta_task * max_path_length
        self.parallel = parallel
        self.total_timesteps_sampled = 0

        # setup vectorized environment
        if self.parallel:
            self.vec_env = MetaParallelEnvExecutor(env, self.meta_batch_size, self.envs_per_task, self.max_path_length)
        else:
            self.vec_env = MetaIterativeEnvExecutor(env, self.meta_batch_size, self.envs_per_task, self.max_path_length)

    def update_tasks(self):
        """
        Samples a new goal for each meta task
        """
        logging.debug("Start update_tasks")
        tasks = self.env.sample_tasks(self.meta_batch_size)
        assert len(tasks) == self.meta_batch_size
        self.vec_env.set_tasks(tasks)
        logging.debug("END update_tasks")
        return tasks

    def obtain_samples(self, log=False, log_prefix=''):
        """
        Collect batch_size trajectories from each task

        Args:
            log (boolean): whether to log sampling times
            log_prefix (str) : prefix for logger

        Returns:
            (dict) : A dict of paths of size [meta_batch_size] x (batch_size) x [5] x (max_path_length)
        """
        logging.debug("Seq2SeqMetaSampler obtain_samples")
        paths = OrderedDict()
        for i in range(self.meta_batch_size):
            paths[i] = []

        n_samples = 0
        running_paths = [_get_empty_running_paths_dict() for _ in range(self.vec_env.num_envs)]
        policy_time, env_time = 0, 0

        policy = self.policy
        obses = self.vec_env.reset()
        while n_samples < self.total_samples:
            t = time.time()
            obs_per_task = np.array(obses)
            actions, logits, values = policy.get_actions(obs_per_task)
            policy_time += time.time() - t
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            env_time += time.time() - t
            new_samples = 0
            for idx, observation, action, logit, reward, value, done, task_finish_times \
                    in zip(itertools.count(), obses, actions, logits, rewards, values, dones, env_infos):
                for single_ob, single_ac, single_logit, single_reward, single_value, single_task_finish_time \
                        in zip(observation, action, logit, reward, value, task_finish_times):
                    running_paths[idx]["observations"] = single_ob
                    running_paths[idx]["actions"] = single_ac
                    running_paths[idx]["logits"] = single_logit
                    running_paths[idx]["rewards"] = single_reward
                    running_paths[idx]["finish_time"] = single_task_finish_time
                    running_paths[idx]["values"] = single_value

                    paths[idx // self.envs_per_task].append(dict(
                        observations=np.squeeze(np.asarray(running_paths[idx]["observations"])),
                        actions=np.squeeze(np.asarray(running_paths[idx]["actions"])),
                        logits = np.squeeze(np.asarray(running_paths[idx]["logits"])),
                        rewards=np.squeeze(np.asarray(running_paths[idx]["rewards"])),
                        finish_time = np.squeeze(np.asarray(running_paths[idx]["finish_time"])),
                        values  = np.squeeze(np.asarray(running_paths[idx]["values"]))
                    ))

                    new_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = _get_empty_running_paths_dict()
            n_samples += new_samples
            obses = next_obses
        self.total_timesteps_sampled += self.total_samples
        if log:
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)
        return paths


def _get_empty_running_paths_dict():
    return dict(observations=[], actions=[], logits=[], rewards=[])




