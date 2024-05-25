import logging
logging.getLogger('tensorflow').disabled = True
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym")
import tensorflow as tf
import numpy as np
import time
from utils import logger
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
class Trainer(object):
    def __init__(self,algo,
                env,
                sampler,
                sample_processor,
                policy,
                n_itr,
                greedy_finish_time,
                start_itr=0,
                inner_batch_size = 500,
                save_interval = 100):
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sampler_processor = sample_processor
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.inner_batch_size = inner_batch_size
        self.greedy_finish_time = greedy_finish_time
        self.save_interval = save_interval
        self.policy_losses= 100000

    def train(self):
        """
        Implement the MRLCO training process for task offloading problem
        """

        start_time = time.time()
        avg_ret = []
        avg_loss = []
        avg_latencies = []
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            # logger.log("\n ---------------- Iteration %d ----------------" % itr)
            # logger.log("Sampling set of tasks/goals for this meta-batch...")

            task_ids = self.sampler.update_tasks()
            paths = self.sampler.obtain_samples(log=False, log_prefix='')

            #print("sampled path length is: ", len(paths[0]))

            greedy_run_time = [self.greedy_finish_time[x] for x in task_ids]
            # logger.logkv('Average greedy latency,', np.mean(greedy_run_time))

            """ ----------------- Processing Samples ---------------------"""
            # logger.log("Processing samples...")
            samples_data = self.sampler_processor.process_samples(paths, log=False, log_prefix='')

            """ ------------------- Inner Policy Update --------------------"""
            policy_losses, value_losses = self.algo.UpdatePPOTarget(samples_data, batch_size=self.inner_batch_size )

            #print("task losses: ", losses)
            # print("average task losses: ", np.mean(policy_losses))
            avg_loss.append(np.mean(policy_losses))
            
            # print("average value losses: ", np.mean(value_losses))

            """ ------------------ Resample from updated sub-task policy ------------"""
            # print("Evaluate the one-step update for sub-task policy")
            new_paths = self.sampler.obtain_samples(log=True, log_prefix='')
            new_samples_data = self.sampler_processor.process_samples(new_paths, log="all", log_prefix='')

            """ ------------------ Outer Policy Update ---------------------"""
            # logger.log("Optimizing policy...")
            self.algo.UpdateMetaPolicy()
            if   np.mean(policy_losses) <  self.policy_losses:
                    self.policy.async_parameters()
                    self.policy_losses = np.mean(policy_losses)

            """ ------------------- Logging Stuff --------------------------"""

            ret = np.array([])
            for i in range(len(new_samples_data)):
                ret = np.concatenate((ret, np.sum(new_samples_data[i]['rewards'], axis=-1)), axis=-1)

            avg_reward = np.mean(ret)

            latency = np.array([])
            for i in range(len(new_samples_data)):
                latency = np.concatenate((latency, new_samples_data[i]['finish_time']), axis=-1)

            avg_latency = np.mean(latency)
            avg_latencies.append(avg_latency)


            # logger.logkv('Itr', itr)
            # logger.logkv('Average reward, ', avg_reward)
            # logger.logkv('Average latency,', avg_latency)
            # logger.logkv("average task losses: ", np.mean(policy_losses))
            # logger.logkv("average value losses: ", np.mean(value_losses))

            # logger.dumpkvs()
            avg_ret.append(avg_reward)

            if itr % self.save_interval == 0:
                self.policy.core_policy.save_variables(save_path="./meta_model_inner_step1/meta_model_"+str(itr)+".ckpt")

        self.policy.core_policy.save_variables(save_path="./meta_model_inner_step1/meta_model_final.ckpt")

        return avg_ret, avg_loss, avg_latencies


class TLBO:
    def __init__(self, population_size, dim, bounds, iterations, trainer):
        self.population_size = population_size
        self.dim = dim
        self.bounds = bounds
        self.iterations = iterations
        self.trainer = trainer
        self.population = np.random.uniform(bounds[:, 0], bounds[:, 1], (population_size, dim))
        self.fitness = np.zeros(population_size)

    def evaluate_population(self):
        for i in range(self.population_size):
            inner_lr, outer_lr = self.population[i]
            self.trainer.algo.inner_lr = inner_lr
            self.trainer.algo.outer_lr = outer_lr

            with tf.compat.v1.Session() as sess:
                sess.run(tf.global_variables_initializer())            
                avg_ret, avg_loss, avg_latencies = self.trainer.train()
            self.fitness[i] = np.mean(avg_loss)  # Assuming we want to minimize the loss

    def teacher_phase(self):
        best_index = np.argmin(self.fitness)
        best_solution = self.population[best_index]
        mean_solution = np.mean(self.population, axis=0)

        for i in range(self.population_size):
            tf = np.random.randint(1, 3)
            new_solution = self.population[i] + np.random.rand(self.dim) * (best_solution - tf * mean_solution)
            new_solution = np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])
            self.population[i] = new_solution

    def learner_phase(self):
        for i in range(self.population_size):
            j = i
            while j == i:
                j = np.random.randint(self.population_size)

            if self.fitness[i] < self.fitness[j]:
                new_solution = self.population[i] + np.random.rand(self.dim) * (self.population[i] - self.population[j])
            else:
                new_solution = self.population[i] + np.random.rand(self.dim) * (self.population[j] - self.population[i])

            new_solution = np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])
            self.population[i] = new_solution

    def optimize(self):
        self.evaluate_population()

        for _ in range(self.iterations):
            self.teacher_phase()
            self.evaluate_population()
            self.learner_phase()
            self.evaluate_population()

        best_index = np.argmin(self.fitness)
        best_solution = self.population[best_index]
        return best_solution


if __name__ == "__main__":
    from env.mec_offloaing_envs.offloading_env import Resources
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
    from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
    from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
    from baselines.vf_baseline import ValueFunctionBaseline
    from meta_algos.MRLCO import MRLCO

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir="./meta_offloading20_log-inner_step1/", format_strs=['stdout', 'log', 'csv'])

    META_BATCH_SIZE = 1

    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=7.0, bandwidth_dl=7.0)

    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=100,
                                graph_number=100,
                                graph_file_paths=[
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20.",
                                    # "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_2/random.20.",
                                    # "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_3/random.20.",
                                    # "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_5/random.20.",
                                    # "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_6/random.20.",
                                    # "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_7/random.20.",
                                    # "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_9/random.20.",
                                    # "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_10/random.20.",
                                    # "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_11/random.20.",
                                    # "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_13/random.20.",
                                    # "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_14/random.20.",
                                    # "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_15/random.20.",
                                    # "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_17/random.20.",
                                    # "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_18/random.20.",
                                    # "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_19/random.20.",
                                    # "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_21/random.20.",
                                    # "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_22/random.20.",
                                    # "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_23/random.20.",
                                    # "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_25/random.20.",
                                ],
                                time_major=False)

    action, greedy_finish_time = env.greedy_solution()
    print("avg greedy solution: ", np.mean(greedy_finish_time))
    print()
    finish_time = env.get_all_mec_execute_time()
    print("avg all remote solution: ", np.mean(finish_time))
    print()
    finish_time = env.get_all_locally_execute_time()
    print("avg all local solution: ", np.mean(finish_time))
    print()

    baseline = ValueFunctionBaseline()
    hparams = tf.contrib.training.HParams(
            unit_type="lstm",
            encoder_units=128,
            decoder_units=128,

            n_features=2,
            time_major=False,
            is_attention=True,
            forget_bias=1.0,
            dropout=0,
            num_gpus=1,
            num_layers=2,
            num_residual_layers=0,
            start_token=0,
            end_token=2,
            is_bidencoder=False
        )
    meta_policy = MetaSeq2SeqPolicy(meta_batch_size=META_BATCH_SIZE, obs_dim=17, encoder_units=128, decoder_units=128,
                                    vocab_size=2, hparams=hparams)

    sampler = Seq2SeqMetaSampler(
        env=env,
        policy=meta_policy,
        rollouts_per_meta_task=1,  # This batch_size is confusing
        meta_batch_size=META_BATCH_SIZE,
        max_path_length=20000,
        parallel=False,
    )

    sample_processor = Seq2SeqMetaSamplerProcessor(baseline=baseline,
                                                   discount=0.99,
                                                   gae_lambda=0.95,
                                                   normalize_adv=True,
                                                   positive_adv=False)
    algo = MRLCO(policy=meta_policy,
                         meta_sampler=sampler,
                         meta_sampler_process=sample_processor,
                         inner_lr=5e-4,
                         outer_lr=5e-4,
                         meta_batch_size=META_BATCH_SIZE,
                         num_inner_grad_steps=1,
                         clip_value = 0.3)

    trainer = Trainer(algo = algo,
                        env=env,
                        sampler=sampler,
                        sample_processor=sample_processor,
                        policy=meta_policy,
                        n_itr=10,
                        greedy_finish_time= greedy_finish_time,
                        start_itr=0,
                        inner_batch_size=1000)

    # with tf.compat.v1.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     avg_ret, avg_loss, avg_latencies = trainer.train()

    bounds = np.array([[1e-5, 1e-2], [1e-5, 1e-2]])
    tlbo = TLBO(population_size=10, dim=2, bounds=bounds, iterations=20, trainer=trainer)

    best_inner_lr, best_outer_lr = tlbo.optimize()
    print(f"Optimal inner_lr: {best_inner_lr}, outer_lr: {best_outer_lr}")
