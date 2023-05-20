import numpy as np
import random
import matplotlib.pyplot as plt
import os
import subprocess
import tensorflow as tf


def generate_dag(n):
    cmd = f'daggen --dot -n {n} --ccr 0.5 --fat 0.5 --regular 0.5 --density 0.6 --mindata 5242880 --maxdata 52428800'
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    lines = output.split('\n')
    edges = []
    for line in lines:
        if '->' in line:
            parts = line.split('->')
            source = int(parts[0].strip())
            target = int(parts[1].split('[')[0].strip())
            if source < n and target < n:
                edges.append((source, target))
    if len(edges) == 0:
        raise ValueError('Failed to generate DAG with any edges')
    dag = np.zeros((n,n))
    for edge in edges:
        dag[edge[0], edge[1]] = 1
    return dag


def mtlbo(dag, task_times, device_times, search_space, rnn_type):
    n_tasks = len(dag)
    n_devices = len(device_times)
    input_shape = (n_tasks, n_devices)

    # Define the neural network architecture
    inputs = tf.keras.Input(shape=input_shape)
    if rnn_type == 'lstm':
        rnn_layer = tf.keras.layers.LSTM(64, activation='relu')(inputs)
    elif rnn_type == 'gru':
        rnn_layer = tf.keras.layers.GRU(64, activation='relu')(inputs)
    else:
        raise ValueError('Invalid RNN type')
    dense1 = tf.keras.layers.Dense(32, activation='relu')(rnn_layer)
    dense2 = tf.keras.layers.Dense(16, activation='relu')(dense1)
    outputs = tf.keras.layers.Dense(n_devices, activation='sigmoid', kernel_constraint=tf.keras.constraints.UnitNorm())(dense2)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')

    # Initialize the population
    pop_size = 10
    pop = np.zeros((pop_size, n_tasks, n_devices))
    for i in range(pop_size):
        pop[i] = np.random.uniform(search_space[0], search_space[1], input_shape)

    # Set the algorithm parameters
    max_iter = 10
    beta = 2
    p = 0.5

    # Evaluate the initial population
    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        fitness[i] = evaluate_solution(pop[i], dag, task_times, device_times, model)

    # Run the main loop
    best_fitness = np.zeros(max_iter)
    best_solution = np.zeros((max_iter, n_tasks, n_devices))
    for t in range(max_iter):
        # Sort the population by fitness
        sorted_indices = np.argsort(fitness)
        pop = pop[sorted_indices]
        fitness = fitness[sorted_indices]

        # Save the best solution
        best_fitness[t] = fitness[0]
        best_solution[t] = pop[0]

        # Update the population
        new_pop = np.zeros((pop_size, n_tasks, n_devices))
        for i in range(pop_size):
            # Select three random solutions
            r1, r2, r3 = random.sample(range(pop_size), 3)

            # Generate a mutant solution
            mutant = pop[r1] + beta * (pop[r2] - pop[r3])

            # Crossover with the current solution
            trial = np.copy(pop[i])
            for j in range(n_tasks):
                for k in range(n_devices):
                    if np.random.rand() < p:
                        trial[j,k] = mutant[j,k]

            # Clip the solution to the search space
            trial = np.clip(trial, search_space[0], search_space[1])

            # Evaluate the trial solution
            trial_fitness = evaluate_solution(trial, dag, task_times, device_times, model)

            # Select the better solution for the next generation
            if trial_fitness < fitness[i]:
                new_pop[i] = trial
                fitness[i] = trial_fitness
            else:
                new_pop[i] = pop[i]

        pop = new_pop

    return best_solution, best_fitness


def evaluate_solution(solution, dag, task_times, device_times, model):
    n_tasks = len(dag)
    n_devices = len(device_times)

    # Calculate the execution time for each task on each device
    task_device_times = np.zeros((n_tasks, n_devices))
    for i in range(n_tasks):
        for j in range(n_devices):
            task_device_times[i,j] = task_times[i] / device_times[j] * solution[i,j]

    # Calculate the critical path time
    c = np.zeros(n_tasks)
    for i in range(n_tasks):
        parents = np.where(dag[:,i] > 0)[0]
        if len(parents) == 0:
            c[i] = task_device_times[i].max()
        else:
            c[i] = task_device_times[i][np.argmax(c[parents])] + task_device_times[i].max()

    return c.max()


def greedy(dag, task_times, device_times):
    n_tasks = len(dag)
    n_devices = len(device_times)

    # Calculate the execution time for each task on each device
    task_device_times = np.zeros((n_tasks, n_devices))
    for i in range(n_tasks):
        for j in range(n_devices):
            task_device_times[i,j] = task_times[i] / device_times[j]

    # Calculate the critical path time
    c = np.zeros(n_tasks)
    for i in range(n_tasks):
        parents = np.where(dag[:,i] > 0)[0]
        if len(parents) == 0:
            c[i] = task_device_times[i].max()
        else:
            c[i] = task_device_times[i][np.argmax(c[parents])] + task_device_times[i].max()

    # Assign tasks to devices greedily
    solution = np.zeros((n_tasks, n_devices))
    tasks = list(range(n_tasks))
    devices = list(range(n_devices))
    while len(tasks) > 0:
        task = tasks.pop(0)
        if len(devices) == 0:
            break
        device = min(devices, key=lambda d: c[task] + task_device_times[task,d])
        solution[task,device] = 1
        devices.remove(device)

    return solution


if __name__ == '__main__':
    n = 40
    dag = generate_dag(n)
    task_times = np.random.uniform(10, 100, n)
    device_times = np.random.uniform(1, 10, 40)
    search_space = (0, 1)
    rnn_type = 'lstm'

    mtlbo_solution_lstm, mtlbo_fitness_lstm = mtlbo(dag, task_times, device_times, search_space, rnn_type)
    rnn_type = 'gru'

    mtlbo_solution, mtlbo_fitness = mtlbo(dag, task_times, device_times, search_space, rnn_type)

    # greedy_solution = greedy(dag, task_times, device_times)
    print(mtlbo_fitness.shape)
    # print(evaluate_solution(greedy_solution, dag, task_times, device_times, None))
    plt.plot(mtlbo_fitness_lstm, label='MTLBO_LSTM')
    plt.plot(mtlbo_fitness, label='MTLBO_GRU')
    # plt.plot(evaluate_solution(greedy_solution, dag, task_times, device_times, None), label='Greedy')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.show()