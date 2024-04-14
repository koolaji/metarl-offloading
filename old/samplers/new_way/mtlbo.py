import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import random

# Function to generate synthetic sine wave data
def generate_sine_data(timesteps=200):
    x = np.linspace(0, 2 * np.pi, timesteps)
    y = np.sin(x)
    return y

# Function to build, train, and evaluate an LSTM model based on given hyperparameters
def train_evaluate_lstm(params):
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

# MTLBO algorithm to optimize hyperparameters
def mtlbo_optimize(objective_func, initial_params, bounds, num_learners=10, generations=20):
    learners = [initial_params.copy() for _ in range(num_learners)]
    best_params = initial_params.copy()
    best_score = float('inf')
    
    for generation in range(generations):
        print(f"Generation {generation+1}/{generations}")
        # Teaching phase
        for index, learner in enumerate(learners):
            score = objective_func(learner)
            print(f"  Learner {index+1}: MSE={score:.15f}, Params={learner}")
            if score < best_score:
                best_score = score
                best_params = learner.copy()
                print(f"  New best score found: {best_score:.15f}")

        # Modifying learner parameters
        for learner in learners:
            for key in learner.keys():
                if random.random() < 0.5:  # Random chance to adjust parameters
                    change = random.uniform(-0.1, 0.1) * (bounds[key][1] - bounds[key][0])
                    learner[key] = np.clip(learner[key] + change, bounds[key][0], bounds[key][1])
            print(f"  Updated Learner Params: {learner}")

    return best_params

# Initial parameters and their bounds
initial_params = {
    'lstm_units': 50,
    'learning_rate': 0.01,
    'batch_size': 20,
    'epochs': 50
}

bounds = {
    'lstm_units': (10, 100),
    'learning_rate': (0.001, 0.01),
    'batch_size': (10, 100),
    'epochs': (10, 100)
}

# Run the optimization
optimized_params = mtlbo_optimize(
    objective_func=train_evaluate_lstm,
    initial_params=initial_params,
    bounds=bounds
)

print("Optimized Parameters:", optimized_params)
