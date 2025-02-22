"""
This class is a subclass of the SampleProcessor class and is specifically designed for processing samples for a
sequence-to-sequence meta-learning algorithm. The main method, process_samples, takes in a dictionary of
paths_meta_batch, where each key is a meta-task and the corresponding value is a list of paths for that task.
The method then goes through each task, fits a baseline estimator using the paths for that task, and computes advantages
 and adjusted rewards. The method then stacks the path data and returns a dictionary of samples_data_meta_batch,
 where each key is a meta-task and the corresponding value is a dictionary of the stacked path data for that task.

The _compute_samples_data method is called by the process_samples method and is responsible for computing discounted
rewards, fitting the baseline estimator, computing advantages, and adjusted rewards. The _append_path_data method is
responsible for appending the path data and returning it as a tuple of numpy arrays.

Overall, this class provides a way to process samples for a sequence-to-sequence meta-learning algorithm by computing
discounted rewards, fitting the baseline estimator, and computing advantages and adjusted rewards.
"""
from samplers.base import SampleProcessor
from utils import utils
import numpy as np


class Seq2SeqMetaSamplerProcessor(SampleProcessor):
    def process_samples(self, paths_meta_batch, log=False, log_prefix=''):
        """
        Processes sampled paths. This involves:
            - computing discounted rewards (returns)
            - fitting baseline estimator using the path returns and predicting the return baselines
            - estimating the advantages using GAE (+ advantage normalization id desired)
            - stacking the path data
            - logging statistics of the paths

        Args:
            paths_meta_batch (dict): A list of dict of lists, size: [meta_batch_size] x (batch_size) x [5] x (max_path_length)
            log (boolean): indicates whether to log
            log_prefix (str): prefix for the logging keys

        Returns:
            (list of dicts) : Processed sample data among the meta-batch; size: [meta_batch_size] x [7] x (batch_size x max_path_length)
        """

        assert isinstance(paths_meta_batch, dict), 'paths must be a dict'
        assert self.baseline, 'baseline must be specified'

        samples_data_meta_batch = []
        all_paths = []

        for meta_task, paths in paths_meta_batch.items():

            # fits baseline, comput advantages and stack path data
            samples_data, paths = self._compute_samples_data(paths)

            samples_data_meta_batch.append(samples_data)
            all_paths.extend(paths)

        # 7) compute normalized trajectory-batch rewards (for E-MAML)
        overall_avg_reward = np.mean(
            np.concatenate([samples_data['rewards'] for samples_data in samples_data_meta_batch]))
        overall_avg_reward_std = np.std(
            np.concatenate([samples_data['rewards'] for samples_data in samples_data_meta_batch]))

        for samples_data in samples_data_meta_batch:
            samples_data['adj_avg_rewards'] = (samples_data['rewards'] - overall_avg_reward) / (
                        overall_avg_reward_std + 1e-8)

        # 8) log statistics if desired
        self._log_path_stats(all_paths, log=log, log_prefix=log_prefix)

        return samples_data_meta_batch

    def _compute_samples_data(self, paths):
        assert type(paths) == list

        # 1) compute discounted rewards (returns)
        for idx, path in enumerate(paths):
            path["returns"] = utils.discount_cumsum(path["rewards"], self.discount)

        # 2) fit baseline estimator using the path returns and predict the return baselines
        self.baseline.fit(paths, target_key="returns")
        all_path_baselines = [self.baseline.predict(path) for path in paths]

        # 3) compute advantages and adjusted rewards
        paths = self._compute_advantages(paths, all_path_baselines)

        observations, actions, logits, rewards, returns, values, advantages, finish_time = self._append_path_data(paths)

        decoder_full_lengths = np.array(observations.shape[0] * [observations.shape[1]])
        # 5) if desired normalize / shift advantages
        if self.normalize_adv:
            advantages = utils.normalize_advantages(advantages)
        if self.positive_adv:
            advantages = utils.shift_advantages_to_positive(advantages)

        # 6) create samples_data object
        samples_data = dict(
            observations=observations,
            decoder_full_lengths=decoder_full_lengths,
            actions=actions,
            logits=logits,
            rewards=rewards,
            returns=returns,
            values=values,
            advantages=advantages,
            finish_time=finish_time
        )

        return samples_data, paths

    def _append_path_data(self, paths):
        observations = np.array([path["observations"] for path in paths])
        actions = np.array([path["actions"] for path in paths])
        logits = np.array([path["logits"] for path in paths])
        rewards = np.array([path["rewards"] for path in paths])
        returns = np.array([path["returns"] for path in paths])
        values = np.array([path["values"] for path in paths])
        advantages = np.array([path["advantages"] for path in paths])
        finish_time = np.array([path["finish_time"] for path in paths])
        
        return observations, actions,logits, rewards, returns, values, advantages, finish_time

