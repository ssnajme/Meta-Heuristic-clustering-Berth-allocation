import random
import numpy as np
import math
import csv
import warnings
from ObjectiveFunctions import cost_calculator
from LocalSearches import LocalSearch
from decimal import Decimal, getcontext
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
import matplotlib.pyplot as plt
from MainNestGeneration import (
    custom_rng,
    num_vessels,
    c_min,
    c_max,
    num_nests,
    a_range,
    b_range,
)
from LocalSearches import LocalSearch



all_costs = []
removed_nests = []  # Initialize an empty list to store removed nests

all_nests = []
num_iterations = 30
weights = [1.1, 1.2, 1.3, 1.5]
penalty_weight = 1.0
T_start = 0
T_end = 100
n = num_vessels

G = 1
A = 0.1
φ = (1 + math.sqrt(5)) / 2
alpha = A / math.sqrt(G)
NumberObjectionEvaluations = n
MaxNumberEvaluations = 15

Lambda = 1.5
dimension = 3
step_size = 0.1

MaxLevyStepSize = 1  # Set your value for MaxLevyStepSize
GoldenRatio = 1.61803398875  # Golden ratio value
n = 10  # Number of nests
#MaxNumberEvaluations = 1000 
MaxNumberEvaluations = 40
G = 1
NumberObjectionEvaluations = 0
top_nests = []

""" 
while NumberObjectionEvaluations < MaxNumberEvaluations and G < MaxNumberEvaluations:
        G += 1
        nests = custom_rng.initialize_nests(num_nests, num_vessels, a_range, b_range, c_min, c_max) 
        for sublist in nests:
            nests.sort(key=lambda x: cost_calculator.calculate_cost_component2(sublist))
            
            for i in range(len(nests) // 2):
                X_i = nests[i]
                alpha = MaxLevyStepSize / math.sqrt(G)
                step_vector = LocalSearch.levy_flight_2(Lambda, dimension, step_size, alpha)
                X_k =[int(coord + step_size * step_item) for coord, step_item in zip(X_i, step_vector)]
                F_i = cost_calculator.calculate_cost_component2(X_i)
                
                if F_i > cost_calculator.calculate_cost_component2(X_k):
                    nests[i] = X_k

            for i in range(len(nests) // 2, len(nests)):
                X_i = nests[i]
                X_j = random.choice(nests[:len(nests) // 2])

                if X_i == X_j:
                    alpha = MaxLevyStepSize / G**2
                    step_vector_2 = LocalSearch.levy_flight_2(Lambda, dimension, step_size, alpha)
                    X_k = [int(coord + step_size * step_item) for coord, step_item in zip(X_i, step_vector)]
                    F_k = cost_calculator.calculate_cost_component2(X_k)
                    l = random.randint(0, len(nests) - 1)

                    if F_k > cost_calculator.calculate_cost_component2(nests[l]):
                        nests[l] = X_k 

                else: 
 
                    nest_x_i = np.array(X_i)
                    #print(nest_x_i)
                    nest_x_j = np.array(X_j)
                    transformed_values = [np.log(abs(float(val)) + 1) for val in nest_x_j]
                    scale_factor = 10
                    scaled_values_j = [int(val * scale_factor) for val in transformed_values]
                    squared_diff = np.sum((np.abs(nest_x_i - scaled_values_j)) ** 2)
                    euclidean_dist = np.sqrt(np.abs(squared_diff)) 
                    dx = int(euclidean_dist / GoldenRatio)
                    X_k = [int(coord + dx) for coord in X_i]
                    F_k = cost_calculator.calculate_cost_component2(X_k)
                    l = random.randint(0, len(nests) - 1)

                    if F_k > cost_calculator.calculate_cost_component2(nests[l]):
                        nests[l] = X_k

        NumberObjectionEvaluations += 1
        
        # Update top_nests
        top_nests = sorted(nests, key=lambda x: cost_calculator.calculate_cost_component2(x))[:30]
        #print(top_nests)
"""

"""
def cuckoo_search_with_cost(MaxNumberEvaluations, num_nests, num_vessels, a_range, b_range, c_min, c_max, cost_calculator, custom_rng, LocalSearch, Lambda, dimension, step_size, MaxLevyStepSize, GoldenRatio):
    NumberObjectionEvaluations = 0
    convergence_costs = []
    G = 0
    while NumberObjectionEvaluations < MaxNumberEvaluations and G < MaxNumberEvaluations:
        G += 1
        nests = custom_rng.initialize_nests(num_nests, num_vessels, a_range, b_range, c_min, c_max)
        for sublist in nests:
            nests.sort(key=lambda x: cost_calculator.calculate_cost_component2(sublist))
            
            for i in range(len(nests) // 2):
                X_i = nests[i]
                alpha = MaxLevyStepSize / math.sqrt(G)
                step_vector = LocalSearch.levy_flight_2(Lambda, dimension, step_size, alpha)
                X_k = [int(coord + step_size * step_item) for coord, step_item in zip(X_i, step_vector)]
                F_i = cost_calculator.calculate_cost_component2(X_i)
                
                if F_i > cost_calculator.calculate_cost_component2(X_k):
                    nests[i] = X_k

            for i in range(len(nests) // 2, len(nests)):
                X_i = nests[i]
                X_j = random.choice(nests[:len(nests) // 2])

                if X_i == X_j:
                    alpha = MaxLevyStepSize / G**2
                    step_vector_2 = LocalSearch.levy_flight_2(Lambda, dimension, step_size, alpha)
                    X_k = [int(coord + step_size * step_item) for coord, step_item in zip(X_i, step_vector)]
                    F_k = cost_calculator.calculate_cost_component2(X_k)
                    l = random.randint(0, len(nests) - 1)

                    if F_k > cost_calculator.calculate_cost_component2(nests[l]):
                        nests[l] = X_k
                else:
                    nest_x_i = np.array(X_i)
                    nest_x_j = np.array(X_j)
                    transformed_values = [np.log(abs(float(val)) + 1) for val in nest_x_j]
                    scale_factor = 10
                    scaled_values_j = [int(val * scale_factor) for val in transformed_values]
                    squared_diff = np.sum((np.abs(nest_x_i - scaled_values_j)) ** 2)
                    euclidean_dist = np.sqrt(np.abs(squared_diff))
                    dx = int(euclidean_dist / GoldenRatio)

                    X_k = [int(coord + dx) for coord in X_i]
                    F_k = cost_calculator.calculate_cost_component2(X_k)
                    l = random.randint(0, len(nests) - 1)

                    if F_k > cost_calculator.calculate_cost_component2(nests[l]):
                        nests[l] = X_k

    
        # Update top_nests based on the distance 
        top_nests = sorted(nests, key=lambda x: cost_calculator.calculate_cost_component2(x))[:30]
        #transformed_values = [np.log(abs(float(val)) + 1) for val in top_nests]
        #transformed_value = np.log(abs(float(top_nests) - min_value) + 1)
        best_cost = cost_calculator.calculate_cost_component2(top_nests[0])
        convergence_costs.append(best_cost)

        NumberObjectionEvaluations += 1

    return top_nests, convergence_costs
""" 

from sklearn.cluster import KMeans

def cuckoo_search_with_cost(MaxNumberEvaluations, num_nests, num_vessels, a_range, b_range, c_min, c_max, cost_calculator, custom_rng, LocalSearch, Lambda, dimension, step_size, MaxLevyStepSize, GoldenRatio):
    NumberObjectionEvaluations = 0
    convergence_costs = []
    G = 0
    while NumberObjectionEvaluations < MaxNumberEvaluations and G < MaxNumberEvaluations:
        G += 1
        nests = custom_rng.initialize_nests(num_nests, num_vessels, a_range, b_range, c_min, c_max)

        for sublist in nests:
            nests.sort(key=lambda x: cost_calculator.calculate_cost_component2(sublist))
            
            for i in range(len(nests) // 2):
                X_i = nests[i]
                alpha = MaxLevyStepSize / math.sqrt(G)
                step_vector = LocalSearch.levy_flight_2(Lambda, dimension, step_size, alpha)
                X_k = [int(coord + step_size * step_item) for coord, step_item in zip(X_i, step_vector)]
                F_i = cost_calculator.calculate_cost_component2(X_i)
                
                if F_i > cost_calculator.calculate_cost_component2(X_k):
                    print("Before updating nests: ", nests[i])
                    nests[i] = X_k
                    print("After updating nests: ", nests[i])

            for i in range(len(nests) // 2, len(nests)):
                X_i = nests[i]
                X_j = random.choice(nests[:len(nests) // 2])

                if X_i == X_j:
                    alpha = MaxLevyStepSize / G**2
                    step_vector_2 = LocalSearch.levy_flight_2(Lambda, dimension, step_size, alpha)
                    X_k = [int(coord + step_size * step_item) for coord, step_item in zip(X_i, step_vector)]
                    F_k = cost_calculator.calculate_cost_component2(X_k)
                    l = random.randint(0, len(nests) - 1)

                    if F_k > cost_calculator.calculate_cost_component2(nests[l]):
                        print("Before updating nests[l]: ", nests[l])
                        nests[l] = X_k
                        print("After updating nests[l]: ", nests[l])
                else:
                    nest_x_i = np.array(X_i)
                    nest_x_j = np.array(X_j)
                    transformed_values = [np.log(abs(float(val)) + 1) for val in nest_x_j]
                    scale_factor = 10
                    scaled_values_j = [int(val * scale_factor) for val in transformed_values]
                    squared_diff = np.sum((np.abs(nest_x_i - scaled_values_j)) ** 2)
                    euclidean_dist = np.sqrt(np.abs(squared_diff))
                    dx = int(euclidean_dist / GoldenRatio)

                    X_k = [int(coord + dx) for coord in X_i]
                    F_k = cost_calculator.calculate_cost_component2(X_k)
                    l = random.randint(0, len(nests) - 1)

                    if F_k > cost_calculator.calculate_cost_component2(nests[l]):
                        nests[l] = X_k

        # Clustering the nests to update top_nests
        X = np.array(nests)
        kmeans = KMeans(n_clusters=30, random_state=0).fit(X)
        cluster_centers = kmeans.cluster_centers_
        distances = [np.linalg.norm(np.array(nest) - cluster_center) for nest in nests for cluster_center in cluster_centers]
        sorted_indices = np.argsort(distances)
        top_nests = [nests[i] for i in sorted_indices if i < len(nests)][:30]

        best_cost = cost_calculator.calculate_cost_component2(top_nests[0])
        convergence_costs.append(best_cost)

        NumberObjectionEvaluations += 1

    return top_nests, convergence_costs


# Example usage:
# Replace the placeholders with actual values for your problem
# result = cuckoo_search_with_cost(MaxNumberEvaluations, num_nests, num_vessels, a_range, b_range, c_min, c_max, cost_calculator, custom_rng, LocalSearch, Lambda, dimension, step_size, MaxLevyStepSize, GoldenRatio)
def plot_convergence(convergence_costs):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(convergence_costs) + 1), convergence_costs, marker='o', linestyle='-', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Convergence Plot')
    plt.grid(True)
    plt.show()

# Example usage:
# Replace the placeholders with actual values for your problem
top_nests, convergence_costs = cuckoo_search_with_cost(MaxNumberEvaluations, num_nests, num_vessels, a_range, b_range, c_min, c_max, cost_calculator, custom_rng, LocalSearch, Lambda, dimension, step_size, MaxLevyStepSize, GoldenRatio)
print(top_nests)
        # Print top nests and their costs
for i, nest in enumerate(top_nests):
    cost = cost_calculator.calculate_cost_component2(nest)
    print(f"Top Nest {i+1}: {nest} (Cost: {cost})")

# Call the plotting function separately
plot_convergence(convergence_costs)

## cpu usage plots#### 
""" 
import psutil
import matplotlib.pyplot as plt
from time import sleep

def monitor_cpu_usage(duration_sec=60):
    cpu_usage = []  # List to store CPU usage values
    interval = 1  # Sampling interval in seconds

    for _ in range(duration_sec):
        cpu_percent = psutil.cpu_percent(interval=interval)
        cpu_usage.append(cpu_percent)
        sleep(interval)

    return cpu_usage

# Example usage:
if __name__ == "__main__":
    duration = 60  # Monitor CPU usage for 60 seconds
    cpu_usage_values = monitor_cpu_usage(duration)

    # Create a simple CPU usage plot
    plt.plot(cpu_usage_values)
    plt.xlabel("Time (seconds)")
    plt.ylabel("CPU Usage (%)")
    plt.title("CPU Usage Over Time")
    plt.grid(True)
    plt.show()
"""
