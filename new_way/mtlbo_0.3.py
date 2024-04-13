import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

def generate_sine_data(timesteps=200):
    """Generate sine wave data."""
    x = np.linspace(0, 2 * np.pi, timesteps)
    y = np.sin(x)
    return y

def objective_function(params):
    """Objective function to evaluate model performance."""
    data = generate_sine_data()
    X = np.array([data[i:i+10] for i in range(len(data)-10)])
    y = np.array([data[i+10] for i in range(len(data)-10)])
    X = np.expand_dims(X, axis=2)

    model = Sequential([
        LSTM(int(params['lstm_units']), activation='relu', input_shape=(10, 1)),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                  loss='mean_squared_error')
    
    model.fit(X, y, epochs=int(params['epochs']), batch_size=int(params['batch_size']), verbose=0)
    mse = model.evaluate(X, y, verbose=0)
    return mse

def mtlbo(population_size, max_generations):
    """MTLBO algorithm to optimize hyperparameters."""
    # Generate initial population
    population = [{'lstm_units': random.randint(20, 100), 
                   'learning_rate': 10**(-random.uniform(2, 4)), 
                   'batch_size': int(2**random.randint(3, 6)), 
                   'epochs': random.randint(30, 100)} for _ in range(population_size)]
    
    for generation in range(max_generations):
        # Teacher phase
        population = sorted(population, key=lambda ind: objective_function(ind))
        best_individual = population[0]
        mean_params = {k: np.mean([p[k] for p in population]) for k in best_individual.keys()}
        
        for individual in population:
            for key in individual.keys():
                individual[key] += np.random.rand() * (best_individual[key] - np.random.rand() * mean_params[key])

        # Learner phase
        for i in range(population_size):
            j = random.randint(0, population_size - 1)
            if i != j:
                if objective_function(population[i]) > objective_function(population[j]):
                    population[i][key] += (population[j][key] - population[i][key]) * np.cos(np.random.rand())
                else:
                    population[i][key] += (population[i][key] - population[j][key]) * np.sin(np.random.rand())

    # Evaluate final population
    final_population = sorted(population, key=lambda ind: objective_function(ind))
    best_params = final_population[0]
    return best_params

# Optimization process
best_hyperparams = mtlbo(population_size=10, max_generations=5)
print("Best Hyperparameters:", best_hyperparams)
