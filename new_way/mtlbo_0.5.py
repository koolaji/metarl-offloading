import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import random
from tqdm import tqdm

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
    data = generate_sine_data()
    X = np.array([data[i:i+10] for i in range(len(data)-10)])
    y = np.array([data[i+10] for i in range(len(data)-10)])
    X = np.expand_dims(X, axis=2)

    model = Sequential([
        LSTM(int(params['lstm_units']), activation='relu', input_shape=(10, 1)),
        Dense(1)
    ])
    # Implement a learning rate scheduler
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=params['learning_rate'],
        decay_steps=100,
        decay_rate=0.96,
        staircase=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss='mean_squared_error')
    
    try:
        model.fit(X, y, epochs=int(params['epochs']), batch_size=int(params['batch_size']), verbose=0)
    except Exception as e:
        print(f"An error occurred during model training: {str(e)}")
        print(f"Parameters: {params}")
        return float('inf')  # Continue optimization with a penalty for this parameter set
    mse = model.evaluate(X, y, verbose=0)
    return mse


def enforce_boundaries(individual):
    """Ensure all parameters stay within their valid range."""
    individual['lstm_units'] = np.clip(individual['lstm_units'], 10, 200)  # At least 10 units, no more than 200
    individual['learning_rate'] = np.clip(individual['learning_rate'], 1e-4, 1e-1)  # Sensible range for learning rate
    individual['batch_size'] = np.clip(individual['batch_size'], 16, 128)  # Batch sizes are typical powers of 2
    individual['epochs'] = np.clip(individual['epochs'], 10, 100)  # At least 10 epochs, no more than 100

def mtlbo(population_size, max_generations):
    """MTLBO algorithm to optimize hyperparameters."""
    # Generate initial population
    population = [{'lstm_units': random.randint(10, 200), 
                   'learning_rate': 10**(-random.uniform(2, 4)), 
                   'batch_size': int(2**random.randint(4, 7)),  # Batch size from 16 to 128
                   'epochs': random.randint(10, 100),
                   'score': float('inf')} for _ in range(population_size)]
    
    for generation in tqdm(range(max_generations), desc='Optimizing Hyperparameters'):
        # Evaluate current population
        for individual in population:
            individual['score'] = objective_function(individual)
        
        # Report the best score so far
        best_score = min([p['score'] for p in population])
        tqdm.write(f"Generation {generation + 1}/{max_generations}, Best Score: {best_score}")
        
        # Teacher phase
        population = sorted(population, key=lambda ind: ind['score'])
        best_individual = population[0]
        mean_params = {k: np.mean([p[k] for p in population if k != 'score']) for k in best_individual.keys() if k != 'score'}
        
        for individual in population:
            for key in mean_params:
                if key != 'score':
                    adjustment = np.random.rand() * (best_individual[key] - mean_params[key])
                    individual[key] += adjustment * (1 if np.random.rand() > 0.5 else -1)
                    enforce_boundaries(individual)  # Enforce boundaries after adjustment
                    individual['score'] = objective_function(individual)  # Re-evaluate after update

        # Learner phase
        for i in range(population_size):
            partner_index = random.randint(0, population_size - 1)
            if i != partner_index:
                partner = population[partner_index]
                if population[i]['score'] > partner['score']:
                    for key in individual:
                        if key != 'score':
                            adjustment = (individual[key] - partner[key]) * np.cos(np.random.rand() * np.pi)
                            individual[key] -= adjustment
                            enforce_boundaries(individual)  # Enforce boundaries after adjustment
                            individual['score'] = objective_function(individual)  # Re-evaluate after update

    # Final evaluation and selection of the best parameters
    final_population = sorted(population, key=lambda ind: ind['score'])
    best_params = final_population[0]
    return best_params

# Optimization process
best_hyperparams = mtlbo(population_size=10, max_generations=10)
print("Best Hyperparameters:", best_hyperparams)
