import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate synthetic sine wave data
def generate_data(timesteps):
    x = np.linspace(0, 2 * np.pi, timesteps)
    y = np.sin(x)
    return y

timesteps = 200
data = generate_data(timesteps)
X = np.array([data[i:i+10] for i in range(len(data)-10)])
y = np.array([data[i+10] for i in range(len(data)-10)])
X = np.expand_dims(X, axis=2)  # Reshape for LSTM [samples, time steps, features]

# Define the LSTM model architecture
model = Sequential([
    LSTM(50, activation='relu', input_shape=(10, 1)),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
epochs = 100
history = model.fit(X, y, epochs=epochs, verbose=1)

# Evaluate the model using the last 20% of the data for testing
test_size = int(0.2 * len(X))
X_test = X[-test_size:]
y_test = y[-test_size:]
predictions = model.predict(X_test)
mse = tf.keras.losses.MeanSquaredError()
test_mse = mse(y_test, predictions).numpy()

print(f"Test MSE: {test_mse}")
