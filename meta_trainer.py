import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import numpy as np
import time
from utils import logger
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym")
from samplers.vectorized_env_executor import  MetaIterativeEnvExecutor


class Trainer(object):
    def __init__(self, 
                 algo,
                 env,
                 sampler,
                 sampler_processor,
                 policy,
                 n_itr,
                 greedy_finish_time,
                 start_itr=0,
                 inner_batch_size=500,
                 save_interval=100):
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sampler_processor = sampler_processor
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.inner_batch_size = inner_batch_size
        self.greedy_finish_time = greedy_finish_time
        self.save_interval = save_interval

    def train(self):
        """
        Implement the MRLCO training process for task offloading problem
        """
        logging.debug('Start train')
        start_time = time.time()
        avg_ret = []
        avg_loss = []
        avg_latencies = []
        for itr in range(self.start_itr, self.n_itr):
            current_time = time.time()
            logging.debug("\n ---------------- Iteration %d ----------------" % itr)
            logging.debug("Sampling set of tasks/goals for this meta-batch...")
            # random select some task and set task_id
            task_ids = self.sampler.update_tasks()
            logging.debug(" task_ids %s", task_ids)
            paths = self.sampler.obtain_samples(log=False, log_prefix='')
            # logging.debug(" paths %s", paths)
            logging.debug("sampled path length is: %s", len(paths[0]))

            greedy_run_time = [self.greedy_finish_time[x] for x in task_ids]
            logger.debug('Average greedy latency,', np.mean(greedy_run_time))
            logger.debug('greedy_run_time,', greedy_run_time)

            """ ----------------- Processing Samples ---------------------"""
            logger.debug("Processing samples...")
            samples_data = self.sampler_processor.process_samples(paths, log=False, log_prefix='')

            """ ------------------- Inner Policy Update --------------------"""
            policy_losses, value_losses = self.algo.UpdatePPOTarget(samples_data, batch_size=self.inner_batch_size )

            print("task losses: ", value_losses)
            print("average task losses: ", np.mean(policy_losses))
            avg_loss.append(np.mean(policy_losses))

            print("average value losses: ", np.mean(value_losses))

            """ ------------------ Resample from updated sub-task policy ------------"""
            print("Evaluate the one-step update for sub-task policy")
            new_paths = self.sampler.obtain_samples(log=True, log_prefix='')
            new_samples_data = self.sampler_processor.process_samples(new_paths, log="all", log_prefix='')

            """ ------------------ Outer Policy Update ---------------------"""
            logger.log("Optimizing policy...")
            self.algo.UpdateMetaPolicy()

            """ ------------------- Logging Stuff --------------------------"""

            ret = np.array([])
            for i in range(1):
                ret = np.concatenate((ret, np.sum(new_samples_data[i]['rewards'], axis=-1)), axis=-1)

            avg_reward = np.mean(ret)

            latency = np.array([])
            for i in range(1):
                latency = np.concatenate((latency, new_samples_data[i]['finish_time']), axis=-1)

            avg_latency = np.mean(latency)
            avg_latencies.append(avg_latency)

            logger.logkv('Itr', itr)
            logger.logkv('Average reward, ', avg_reward)
            logger.logkv('Average latency,', "{:.4f}".format(avg_latency))

            logger.dumpkvs()
            avg_ret.append(avg_reward)

            if itr % self.save_interval == 0:
                self.policy.core_policy.save_variables(save_path="./meta_model_inner_step1/meta_model_"+str(itr)+".ckpt")
            elapsed_time = time.time() - current_time
            logging.debug("Elapsed time: %s  seconds ", str(elapsed_time))

        self.policy.core_policy.save_variables(save_path="./meta_model_inner_step1/meta_model_final.ckpt")
        elapsed_time = time.time() - start_time
        print("Elapsed time:", elapsed_time, "seconds")
        return avg_ret, avg_loss, avg_latencies


if __name__ == "__main__":
    from env.mec_offloaing_envs.offloading_env import Resources  # import resource spec
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment  # import env spec
    from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
    from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
    from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
    from baselines.vf_baseline import ValueFunctionBaseline
    from meta_algos.MRLCO import MRLCO
    from meta_algos.TLBO import TLBO
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logging.basicConfig(level=logging.DEBUG, filename='meta_train.log',  filemode='a',)
    logging.root.setLevel(logging.ERROR)
    logger.configure(dir="./meta_offloading20_log-inner_step1/", format_strs=['stdout', 'log', 'csv'])
    META_BATCH_SIZE = 1
    logging.debug('starting')
    
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
    logging.debug('start of greedy_solution')
    action, greedy_finish_time = env.greedy_solution()
    logging.debug('end of greedy_solution')
    logging.debug("avg greedy solution: %s", np.mean(greedy_finish_time))
    finish_time = env.get_all_mec_execute_time()
    logging.debug("avg all remote solution: %s", np.mean(finish_time))
    finish_time = env.get_all_locally_execute_time()
    logging.debug("avg all local solution: %s", np.mean(finish_time))

    baseline = ValueFunctionBaseline()

    meta_policy = MetaSeq2SeqPolicy(meta_batch_size=META_BATCH_SIZE, obs_dim=17, encoder_units=128, decoder_units=128,
                                    vocab_size=2)
    
    # vec_env = MetaIterativeEnvExecutor(env, meta_batch_size=META_BATCH_SIZE, envs_per_task=1, max_path_length=20000)
    # obses = vec_env.reset()
    sampler = Seq2SeqMetaSampler(
        env=env,
        policy=meta_policy,
        rollouts_per_meta_task=1,  # This batch_size is confusing
        meta_batch_size=META_BATCH_SIZE,
        max_path_length=20000,
        parallel=True,
    )
    sampler_processor = Seq2SeqMetaSamplerProcessor(baseline=baseline,
                                                   discount=0.99,
                                                   gae_lambda=0.95,
                                                   normalize_adv=True,
                                                   positive_adv=False)

    
    algo = MRLCO(policy=meta_policy,
                 meta_sampler=sampler,
                 meta_sampler_process=sampler_processor,
                 inner_lr=5e-4,
                 outer_lr=5e-4,
                 meta_batch_size=META_BATCH_SIZE,
                 num_inner_grad_steps=1,
                 clip_value=0.3)
    
    # tlbo = TLBO (population_size = 100, 
    #              num_generations =100, 
    #              policy=meta_policy, 
    #              env=env, 
    #              sampler=sampler, 
    #              sampler_processor=sampler_processor)

    trainer = Trainer(
                      algo=algo,
                      env=env,
                      sampler=sampler,
                      sampler_processor=sampler_processor,
                      policy=meta_policy,
                      n_itr=25,
                      greedy_finish_time= greedy_finish_time,
                      start_itr=0,
                      inner_batch_size=1000)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #### test ####
        # obs_per_task = np.array(obses)
        # actions, logits, values = meta_policy.get_actions(obs_per_task)    
        # next_obses, rewards, dones, env_infos = vec_env.step(actions)
        # next_obs_per_task = np.array(next_obses)
        # diff_array = np.setdiff1d(next_obs_per_task , obs_per_task)
        # logging.debug('diff_array %s',diff_array)
        ####

        avg_ret, avg_loss, avg_latencies = trainer.train()
        # trainer.train()
        logging.info("final result %s, %s, %s ", avg_ret, avg_loss, avg_latencies)


