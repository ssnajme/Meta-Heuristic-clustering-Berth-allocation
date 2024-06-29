import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the generator model
def build_generator(latent_dim, num_vessels):
    model = keras.Sequential([
        keras.Input(shape=(latent_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_vessels * 7)  # Output should be 7 times the number of vessels for C2, C3, C4, ETA, At, Dt, d
    ])
    return model

# Generate random latent vectors
def generate_latent_points(latent_dim, num_samples):
    return np.random.normal(0, 1, (num_samples, latent_dim))

# Generate nests using the generator model with scaled integer values
def generate_nests_scaled_integer(generator, latent_dim, num_nests, num_vessels, scale_factor=10):
    latent_points = generate_latent_points(latent_dim, num_nests)
    nests = generator.predict(latent_points)
    scaled_nests = nests * scale_factor  # Scale the values
    rounded_nests = np.rint(scaled_nests)  # Round the scaled values to the nearest integer
    return rounded_nests.reshape(num_nests, num_vessels, 7)  # Reshape to have 7 values per vessel

# Parameters for generating nests
num_nests = 10
num_vessels = 5
latent_dim = 100

# Initialize the generator model
generator = build_generator(latent_dim, num_vessels)

# Generate nests using the generator model
generated_nests = generate_nests_scaled_integer(generator, latent_dim, num_nests, num_vessels)

print(generated_nests)