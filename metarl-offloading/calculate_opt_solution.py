import numpy as np

from env.mec_offloaing_envs.offloading_env import Resources
from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
from utils import utils

if __name__ == "__main__":
    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=5.5, bandwidth_dl=5.5)

    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=100,
                                graph_number=100,
                                graph_file_paths=[
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_23/random.20."
                                ],
                                time_major=False)

    env.set_task(0)

    print("calculate optimal solution offloading20_23======")
    cost, plan = env.calculate_optimal_solution()
    target_batch, task_finish_time_batch = env.get_reward_batch_step_by_step(plan,
                                                                             env.task_graphs_batchs[env.task_id],
                                                                             env.max_running_time_batchs[env.task_id],
                                                                             env.min_running_time_batchs[env.task_id])
    discounted_reward = []
    for reward_path in target_batch:
        discounted_reward.append(utils.discount_cumsum(reward_path, 1.0)[0])

    print("avg optimal solution: ", np.mean(discounted_reward))
    print("avg optimal solution: ", np.mean(task_finish_time_batch))
    print("avg optimal solution: ", np.mean(cost))

    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=5.5, bandwidth_dl=5.5)

    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=100,
                                graph_number=100,
                                graph_file_paths=[
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_17/random.20."
                                ],
                                time_major=False)

    env.set_task(0)

    print("calculate optimal solution offloading20_17======")
    cost, plan = env.calculate_optimal_solution()
    target_batch, task_finish_time_batch = env.get_reward_batch_step_by_step(plan,
                                                                             env.task_graphs_batchs[env.task_id],
                                                                             env.max_running_time_batchs[env.task_id],
                                                                             env.min_running_time_batchs[env.task_id])
    discounted_reward = []
    for reward_path in target_batch:
        discounted_reward.append(utils.discount_cumsum(reward_path, 1.0)[0])

    print("avg optimal solution: ", np.mean(discounted_reward))
    print("avg optimal solution: ", np.mean(task_finish_time_batch))
    print("avg optimal solution: ", np.mean(cost))

    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=5.5, bandwidth_dl=5.5)

    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=100,
                                graph_number=100,
                                graph_file_paths=[
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_18/random.20."
                                ],
                                time_major=False)

    env.set_task(0)

    print("calculate optimal solution offloading20_18======")
    cost, plan = env.calculate_optimal_solution()
    target_batch, task_finish_time_batch = env.get_reward_batch_step_by_step(plan,
                                                                             env.task_graphs_batchs[env.task_id],
                                                                             env.max_running_time_batchs[env.task_id],
                                                                             env.min_running_time_batchs[env.task_id])
    discounted_reward = []
    for reward_path in target_batch:
        discounted_reward.append(utils.discount_cumsum(reward_path, 1.0)[0])

    print("avg optimal solution: ", np.mean(discounted_reward))
    print("avg optimal solution: ", np.mean(task_finish_time_batch))
    print("avg optimal solution: ", np.mean(cost))

    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=5.5, bandwidth_dl=5.5)

    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=100,
                                graph_number=100,
                                graph_file_paths=[
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_19/random.20."
                                ],
                                time_major=False)

    env.set_task(0)

    print("calculate optimal solution offloading20_19======")
    cost, plan = env.calculate_optimal_solution()
    target_batch, task_finish_time_batch = env.get_reward_batch_step_by_step(plan,
                                                                             env.task_graphs_batchs[env.task_id],
                                                                             env.max_running_time_batchs[env.task_id],
                                                                             env.min_running_time_batchs[env.task_id])
    discounted_reward = []
    for reward_path in target_batch:
        discounted_reward.append(utils.discount_cumsum(reward_path, 1.0)[0])

    print("avg optimal solution: ", np.mean(discounted_reward))
    print("avg optimal solution: ", np.mean(task_finish_time_batch))
    print("avg optimal solution: ", np.mean(cost))

    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=5.5, bandwidth_dl=5.5)

    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=100,
                                graph_number=100,
                                graph_file_paths=[
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_21/random.20."
                                ],
                                time_major=False)

    env.set_task(0)

    print("calculate optimal solution offloading20_21======")
    cost, plan = env.calculate_optimal_solution()
    target_batch, task_finish_time_batch = env.get_reward_batch_step_by_step(plan,
                                                                             env.task_graphs_batchs[env.task_id],
                                                                             env.max_running_time_batchs[env.task_id],
                                                                             env.min_running_time_batchs[env.task_id])
    discounted_reward = []
    for reward_path in target_batch:
        discounted_reward.append(utils.discount_cumsum(reward_path, 1.0)[0])

    print("avg optimal solution: ", np.mean(discounted_reward))
    print("avg optimal solution: ", np.mean(task_finish_time_batch))
    print("avg optimal solution: ", np.mean(cost))

    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=5.5, bandwidth_dl=5.5)

    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=100,
                                graph_number=100,
                                graph_file_paths=[
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_22/random.20."
                                ],
                                time_major=False)

    env.set_task(0)

    print("calculate optimal solution offloading20_22======")
    cost, plan = env.calculate_optimal_solution()
    target_batch, task_finish_time_batch = env.get_reward_batch_step_by_step(plan,
                                                                             env.task_graphs_batchs[env.task_id],
                                                                             env.max_running_time_batchs[env.task_id],
                                                                             env.min_running_time_batchs[env.task_id])
    discounted_reward = []
    for reward_path in target_batch:
        discounted_reward.append(utils.discount_cumsum(reward_path, 1.0)[0])

    print("avg optimal solution: ", np.mean(discounted_reward))
    print("avg optimal solution: ", np.mean(task_finish_time_batch))
    print("avg optimal solution: ", np.mean(cost))
