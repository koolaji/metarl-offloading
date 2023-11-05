import logging
logging.getLogger('tensorflow').disabled = True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import time
from utils import logger
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym")
from samplers.vectorized_env_executor import  MetaIterativeEnvExecutor
import gc
class Trainer(object):
    def __init__(self, 
                 tlbo,
                 env,
                 sampler,
                 sampler_processor,
                 policy,
                 n_itr,
                 greedy_finish_time,
                 start_itr=0,
                 inner_batch_size=500,
                 save_interval=10,
                 batch_size=10):
        self.tlbo = tlbo
        self.env = env
        self.sampler = sampler
        self.sampler_processor = sampler_processor
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.inner_batch_size = inner_batch_size
        self.greedy_finish_time = greedy_finish_time
        self.save_interval = save_interval
        self.batch_size = batch_size
    def train(self, sess):
        """
        Implement the TLBO training process for task offloading problem
        """
        logging.debug('Start train')
        avg_ret = []
        avg_loss = []
        avg_latencies = []

        self.population = []

        for itr in range(self.start_itr, self.n_itr):
            gc.collect()
            self.population = []
            task_ids = self.sampler.update_tasks()
            logging.info(" task_ids %s", task_ids)
            paths = self.sampler.obtain_samples(log=False, log_prefix='')
            logging.info("sampled path length is: %s", len(paths[0]))

            # """ ----------------- Processing Samples ---------------------"""
            # logger.info("Processing samples...")
            self.samples_data = self.sampler_processor.process_samples(paths, log=False, log_prefix='')
            for _ in range(self.batch_size-1):
                random_params = self.policy.get_params(sess, index=_ )
                self.population.append(random_params)
            logging.debug("\n ---------------- Iteration %d ----------------" % itr)
            logging.debug("Sampling set of tasks/goals for this meta-batch...")
            # logging.info("Sampling set of tasks/goals for this meta-batch...")
            greedy_run_time = [self.greedy_finish_time[x] for x in task_ids[:self.batch_size-1]]
            logger.logkv('Average greedy latency,', np.mean(greedy_run_time))
            for i in range(10):
                tlbo.teacher_phase(population=self.population, iteration=i, max_iterations=10, sess=sess)
                tlbo.learner_phase(population=self.population, iteration=i, max_iterations=10, sess=sess)   
                if (not tlbo.change) and tlbo.teacher != []:
                    break
            """ ------------------- Logging Stuff --------------------------"""
            paths = self.sampler.obtain_samples(log=False, log_prefix='')
            new_samples_data = self.sampler_processor.process_samples(paths, log="all", log_prefix='')
            """ ------------------- Logging Stuff --------------------------"""

            ret = np.array([])
            for i in range(len(new_samples_data)-1):
                ret = np.concatenate((ret, np.sum(new_samples_data[i]['rewards'], axis=-1)), axis=-1)

            avg_reward = np.mean(ret)

            latency = np.array([])
            for i in range(len(new_samples_data)-1):
                latency = np.concatenate((latency, new_samples_data[i]['finish_time']), axis=-1)

            avg_latency = np.mean(latency)
            avg_latencies.append(avg_latency)


            logger.logkv('Itr', itr)
            logger.logkv('Average reward, ', avg_reward)
            logger.logkv('Average latency,', avg_latency)
            logger.dumpkvs()
            avg_ret.append(np.mean(avg_reward))

            if itr % self.save_interval == 0:
                self.policy.set_params_core(tlbo.teacher, sess)
                self.policy.core_policy.save_variables(save_path="./meta_model_inner_step1/meta_model_"+str(itr)+".ckpt")

        self.policy.core_policy.save_variables(save_path="./meta_model_inner_step1/meta_model_final.ckpt")
        return avg_ret, avg_loss, avg_latencies


if __name__ == "__main__":
    from env.mec_offloaing_envs.offloading_env import Resources  # import resource spec
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment  # import env spec
    from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
    from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
    from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
    from baselines.vf_baseline import ValueFunctionBaseline
    # from meta_algos.MRLCO import MRLCO
    from meta_algos.MTLBO import MTLBO
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    logging.basicConfig(level=logging.INFO, filename='meta_train.log',  filemode='a',)
    logging.root.setLevel(logging.INFO)
    logger.configure(dir="./meta_offloading20_log-inner_step1/", format_strs=['stdout', 'log', 'csv'])
    META_BATCH_SIZE = 5
    logging.debug('starting')
    
    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=7.0, bandwidth_dl=7.0)

    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=100,
                                graph_number=100,
                                graph_file_paths=[
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_2/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_3/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_5/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_6/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_7/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_9/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_10/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_11/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_13/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_14/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_15/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_17/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_18/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_19/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_21/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_22/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_23/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_25/random.20.",
                                ],
                                time_major=False)
    logging.info('start of greedy_solution')
    action, greedy_finish_time = env.greedy_solution()
    logging.info('end of greedy_solution')
    logging.info("avg greedy solution: %s", np.mean(greedy_finish_time))
    finish_time = env.get_all_mec_execute_time()
    logging.info("avg all remote solution: %s", np.mean(finish_time))
    finish_time = env.get_all_locally_execute_time()
    logging.info("avg all local solution: %s", np.mean(finish_time))

    logging.info('baseline')
    baseline = ValueFunctionBaseline()
    logging.info('meta_policy')
    meta_policy = MetaSeq2SeqPolicy(meta_batch_size=META_BATCH_SIZE, obs_dim=17, encoder_units=128, decoder_units=128,
                                    vocab_size=2)
    logging.info('sampler')
    sampler = Seq2SeqMetaSampler(
        env=env,
        policy=meta_policy,
        rollouts_per_meta_task=1,  
        meta_batch_size=META_BATCH_SIZE,
        max_path_length=20000,
        parallel=True,
    )
    logging.info('sampler_processor')
    sampler_processor = Seq2SeqMetaSamplerProcessor(baseline=baseline,
                                                   discount=0.99,
                                                   gae_lambda=0.95,
                                                   normalize_adv=True,
                                                   positive_adv=False)

    
    logging.info('tlbo')
    tlbo = MTLBO (policy=meta_policy, 
                 env=env, 
                 sampler=sampler, 
                 sampler_processor=sampler_processor,
                 batch_size=META_BATCH_SIZE,
                 inner_batch_size=1000,
                 population_index=[])
    logging.info('trainer')
    trainer = Trainer(
                      tlbo=tlbo,
                      env=env,
                      sampler=sampler,
                      sampler_processor=sampler_processor,
                      policy=meta_policy,
                      n_itr=2000,
                      greedy_finish_time= greedy_finish_time,
                      start_itr=0,
                      inner_batch_size=1000,
                      batch_size=META_BATCH_SIZE)

    with tf.device('/device:XLA_GPU:0'):
      with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        avg_ret, avg_loss, avg_latencies = trainer.train(sess)
        logging.debug("final result %s, %s, %s ", avg_ret, avg_loss, avg_latencies)


