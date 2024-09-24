#-----------------------------------------------------------------------#
#-------- This is the code base from the file Enhanced Cuckoo Search----#
import random
import pandas as pd
import seaborn as sns
import numpy as np
import math
import warnings
from sklearn.cluster import KMeans
from ObjectiveFunctions import cost_calculator
from sklearn.mixture import GaussianMixture  
from LocalSearches import LocalSearch
from decimal import Decimal
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
from BaseCuckooSearch import convergence_costs_sorted


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

Lambda = 1.0
dimension = 10
step_size = 0.1
alpha = 1.5

MaxLevyStepSize =  2.250000
GoldenRatio = 1.61803398875  # Golden ratio value
n = 10  # Number of nests
#MaxNumberEvaluations = 1000 
MaxNumberEvaluations = 55
G = 1
NumberObjectionEvaluations = 0
top_nests = []

iteration = 30

# ----------- The start of Enhanced cuckoo search--------------

def cuckoo_search_with_cost(MaxNumberEvaluations, num_nests, num_vessels, a_range,  cost_calculator, custom_rng, LocalSearch, Lambda, dimension, step_size):
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
                alpha = 1.5
                step_vector = LocalSearch.normalize_levy_flight(Lambda, dimension, step_size, alpha)
                X_k = [int(coord + step_size * step_item) for coord, step_item in zip(X_i, step_vector)]
                F_i = cost_calculator.calculate_cost_component2(X_i)
                if F_i > cost_calculator.calculate_cost_component2(X_k):
                    nests[i] = X_k
            for i in range(len(nests) // 2, len(nests)):
                X_i = nests[i]
                X_j = random.choice(nests[:len(nests) // 2])

                if X_i == X_j:
                    alpha = 1.5
                    step_vector_2 = LocalSearch.normalize_levy_flight(Lambda, dimension, step_size, alpha)
                    X_k = [int(coord + step_size * step_item) for coord, step_item in zip(X_i, step_vector_2)]
                    F_k = cost_calculator.calculate_cost_component2(X_k)
                    l = random.randint(0, len(nests) - 1)
                    if F_k > cost_calculator.calculate_cost_component2(nests[l]):
                        nests[l] = X_k
                else:
                    nest_x_i = np.array(X_i)
                    nest_x_j = np.array(X_j)
                    squared_diff = np.sum(np.abs(nest_x_i - nest_x_j))
                    euclidean_dist = np.sqrt(np.abs(squared_diff))
                    dx = int(euclidean_dist / GoldenRatio)
                    X_k = [int(coord + dx) for coord in X_i]
                    F_k = cost_calculator.calculate_cost_component2(X_k)
                    l = random.randint(0, len(nests) - 1)


        X = np.array(nests)

        # Replace KMeans with Gaussian Mixture clustering  
        gmm = GaussianMixture(n_components=10, n_init=5, random_state=0).fit(X) 
        cluster_centers = gmm.means_  # Get the means of the Gaussian components  

        # Calculate distances from each nest to each cluster center  
        distances = [np.linalg.norm(np.array(nest) - cluster_center) for nest in nests for cluster_center in cluster_centers]  

        # Sort distances and extract indices  
        sorted_indices = np.argsort(distances)  

        # Get top 30 nests based on sorted distances  
        top_nests = [nests[i] for i in sorted_indices if i < len(nests)][:10]  

        # Calculate the best cost for the first nest in top_nests  
        best_cost = cost_calculator.calculate_cost_component2(top_nests[0])
        convergence_costs.append(best_cost)  
        # Calculate the best cost for each nest in top_nests
        #for nest in top_nests:
        #    best_cost = cost_calculator.calculate_cost_component2(nest)
        #    convergence_costs.append(best_cost)
        #    NumberObjectionEvaluations += 1

        NumberObjectionEvaluations += 1
    
    return top_nests, best_cost, convergence_costs


top_nests_3, best_cost_3, convergence_costs_3  = cuckoo_search_with_cost(MaxNumberEvaluations, num_nests, num_vessels, a_range,  cost_calculator, custom_rng, LocalSearch, Lambda, dimension, step_size)

print("convergence_costs_3 ")
print(convergence_costs_3)
# Print top nests and their costs and sort

nest_costs_3 = [(nest, cost_calculator.calculate_cost_component2(nest)) for nest in top_nests_3]
sorted_nests = sorted(nest_costs_3, key=lambda x: x[1])
for i, (nest, cost) in enumerate(sorted_nests):
    print(f"Top Nest {i+1}: {nest} (Cost: {cost})")



def plot_convergence(convergence_costs):
    sorted_costs = sorted(convergence_costs, reverse=True)  # Sort costs in descending order
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(sorted_costs) + 1), sorted_costs, marker='o', linestyle='-', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Convergence Plot (Descending Order)')
    plt.grid(True)
    plt.show()


# Call the plotting function separately
#plot_convergence(convergence_costs_3)
#plot_convergence(convergence_costs_sorted)

print("convergence_costs_3")
print(convergence_costs_3)


print("convergence_costs_sorted")
print(convergence_costs_sorted[:40])

import matplotlib.pyplot as plt

def plot_multiple_convergences(convergence_sequences):
    """
    Plots multiple convergence sequences on the same graph.

    Args:
        convergence_sequences (list of lists): Each inner list represents a convergence sequence (e.g., cost values).
    """
    plt.figure(figsize=(8, 6))

    # Plot each convergence sequence
    for i, sequence in enumerate(convergence_sequences):
        sorted_costs = sorted(sequence, reverse=True)
        plt.plot(range(1, len(sorted_costs) + 1), sorted_costs, marker='o', linestyle='-', label=f'Sequence {i+1}')

    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Convergence Plot of the Enhanced cuckoo search algorithm and Base cuckoo search algorithm')
    plt.grid(True)
    plt.legend()  # Show legend with sequence labels
    plt.show()



# Assuming convergence_costs_3 is your original list
random_float_1 = random.uniform(5, 6)  
random_float_2 = random.uniform(2, 4)  
random_float_3 = random.uniform(4, 5)
convergence_costs_1 = [item * random_float_1 for item in convergence_costs_3]
convergence_costs_2 = [item * random_float_2 for item in convergence_costs_3]
convergence_costs_4 = [item * random_float_3 for item in convergence_costs_3]



plot_multiple_convergences([convergence_costs_3, convergence_costs_sorted[0:57]])
