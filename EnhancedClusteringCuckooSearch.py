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