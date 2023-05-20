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
    dag = np.zeros((n,n))
    for edge in edges:
        dag[edge[0], edge[1]] = 1
    print(dag)
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
    outputs = tf.keras.layers.Dense(n_devices, activation='sigmoid')(dense2)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')

    # Initialize the population
    pop_size = 10
    pop = np.random.uniform(low=search_space[0], high=search_space[1], size=(pop_size, n_tasks, n_devices))

    # Evaluate the fitness of each solution in the population
    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        makespan = evaluate_solution(pop[i], dag, task_times, device_times)
        fitness[i] = 1 / makespan

    # Run the optimization loop
    n_generations = 50
    for generation in range(n_generations):
        # Sort the population by fitness
        sorted_indices = np.argsort(fitness)
        pop = pop[sorted_indices]
        fitness = fitness[sorted_indices]

        # Select the best solution
        best_solution = pop[-1]

        # Generate new solutions using MTLBO
        new_pop = np.zeros((pop_size, n_tasks, n_devices))
        for i in range(pop_size):
            # Select three random solutions from the population
            r1, r2, r3 = np.random.choice(pop[0:pop_size-1], size=3, replace=False)
            donor = r1 + 0.5 * (r2 - r3)

            # Perform crossover between the donor and the current solution
            crossover_prob = np.random.uniform()
            if crossover_prob < 0.7:
                trial = np.where(np.random.rand(n_tasks, n_devices) < 0.5, donor, pop[i])
            else:
                trial = donor

            # Clip the trial solution to the search space
            trial = np.clip(trial, search_space[0], search_space[1])

            # Evaluate the fitness of the trial solution
            trial_fitness = 1 / evaluate_solution(trial, dag, task_times, device_times)

            # Perform selection between the trial solution and the current solution
            if trial_fitness > fitness[i]:
                new_pop[i] = trial
                fitness[i] = trial_fitness
            else:
                new_pop[i] = pop[i]

        # Replace the population with the new population
        pop = new_pop

        # Print the best fitness in the population
        print(f'Generation {generation+1}, best fitness: {fitness[-1]}')

    # Return the best solution
    return best_solution


def evaluate_solution(solution, dag, task_times, device_times):
    n_tasks, n_devices = solution.shape

    # Compute the start time of each task
    start_times = np.zeros(n_tasks)
    for i in range(n_tasks):
        parents = np.where(dag[:,i] == 1)[0]
        if len(parents) == 0:
            start_times[i] = 0
        else:
            max_parent_time = np.max(start_times[parents])
            start_times[i] = max_parent_time + np.max(solution[i] * task_times[i])

    # Compute the makespan of the solution
    end_times = start_times + np.max(solution * task_times, axis=0)
    device_times = np.repeat(device_times[np.newaxis,:], n_tasks, axis=0)
    device_end_times = np.zeros(n_devices)
    for i in range(n_tasks):
        device_start_times = np.maximum(device_end_times, start_times[i])
        device_end_times = device_start_times + solution[i] * task_times[i]
    makespan = np.max(device_end_times)

    return makespan

# Set the parameters
n_tasks = 10
n_devices = 3
task_times = np.random.uniform(low=1, high=10, size=n_tasks)
device_times = np.random.uniform(low=1, high=10, size=n_devices)
search_space = (0, 1)
rnn_type = 'lstm'

# Generate a random DAG
dag = generate_dag(n_tasks)

# Run the MTLBO algorithm
best_solution = mtlbo(dag, task_times, device_times, search_space, rnn_type)

# Print the best solution and its makespan
print('Best solution:')
print(best_solution)
makespan = evaluate_solution(best_solution, dag, task_times, device_times)
print(f'Makespan: {makespan}')