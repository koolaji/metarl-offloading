import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym")

if __name__ == "__main__":
    from env.mec_offloaing_envs.offloading_env import Resources  # import resource spec
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment  # import env spec
    from policies.meta_seq2seq_policy import  Seq2SeqPolicy
    from baselines.vf_baseline import ValueFunctionBaseline
    from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
    from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler


    META_BATCH_SIZE = 2    
    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=7.0, bandwidth_dl=7.0)
    logging.root.setLevel(logging.INFO)
    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=100,
                                graph_number=100,
                                graph_file_paths=[
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_2/random.20.",
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
    logging.info('start of greedy_solution')
    action, greedy_finish_time = env.greedy_solution()
    logging.info('end of greedy_solution')
    logging.info("avg greedy solution: %s", np.mean(greedy_finish_time))
    finish_time = env.get_all_mec_execute_time()
    logging.info("avg all remote solution: %s", np.mean(finish_time))
    finish_time = env.get_all_locally_execute_time()
    logging.info("avg all local solution: %s", np.mean(finish_time))

    baseline = ValueFunctionBaseline()

    meta_policy = MetaSeq2SeqPolicy(meta_batch_size=META_BATCH_SIZE, obs_dim=17, encoder_units=128, decoder_units=128,
                                    vocab_size=2)

    # sampler = Seq2SeqMetaSampler(
    #     env=env,
    #     policy=meta_policy,
    #     rollouts_per_meta_task=1,  # This batch_size is confusing
    #     meta_batch_size=META_BATCH_SIZE,
    #     max_path_length=20000,
    #     parallel=False,
    # )   