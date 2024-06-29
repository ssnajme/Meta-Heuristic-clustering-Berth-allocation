import numpy as np
import math

class LocalSearch:
    @classmethod
    def levy_flight_1(cls, Lambda, dimension, step_size):
        sigma = (
            math.gamma(1 + Lambda)
            * np.sin(np.pi * Lambda / 2)
            / (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))
        ) ** (1 / Lambda)
        u = np.random.normal(0, sigma, size=dimension)
        v = np.random.normal(0, 1, size=dimension)
        step = u / np.abs(v) ** (1 / Lambda)
        step = step_size * step
        return step

    @classmethod
    def levy_flight_2(cls, Lambda, dimension, step_size, alpha):
        sigma = ((math.gamma(1 + alpha) * np.sin(np.pi * alpha / 2)) / (math.gamma((1 + alpha) / 2) * alpha * 2 ** ((alpha - 1) / 2))) ** (1 / alpha)
        u = np.random.normal(0, sigma, dimension)
        v = np.random.normal(0, 1, dimension)
        step = u / abs(v) ** (1 / alpha)
        
        # Generate a list with 7 values by repeating the step vector
        step_vector = [step[i % dimension] for i in range(7)]
        
        return step_vector
    
    def brownian_walk(num_items):
        increments = np.random.normal(0, 1, num_items)
        walk = np.cumsum(increments)
        return walk


# Example of using the LocalSearch class
Lambda = 1.5
dimension = 3
step_size = 0.1
alpha = 2.0

step_1 = LocalSearch.levy_flight_1(Lambda, dimension, step_size)
step_2 = LocalSearch.levy_flight_2(Lambda, dimension, step_size, alpha)

#print("Step 1:", step_1)
#print("Step 2:", step_2)