import numpy as np
import csv
import random
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from ObjectiveFunctions import cost_calculator
from LocalSearches import LocalSearch
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

all_nests = []
all_cost = []
nests = []
removed_nests = []
top_nests = []
top_nests_with_costs = []


num_iterations = 30
weights = [1.1, 1.2, 1.3, 1.5]
penalty_weight = 1.0
T_start = 0
T_end = 100

Lambda = 1.5
dimension = 3
step_size = 0.1


def optimize_nests(num_iterations, num_nests, num_vessels, a_range, b_range, c_min, c_max, Lambda, step_size):
    """
            Optimizes nests using the specified parameters.
        Args:
            num_iterations (int): Number of iterations.
            num_nests (int): Number of nests.
            num_vessels (int): Number of vessels.
            a_range (float): Lower bound for nest initialization.
            b_range (float): Upper bound for nest initialization.
            c_min (float): Minimum value for nest initialization.
            c_max (float): Maximum value for nest initialization.
            Lambda (float): Levy flight parameter.
            step_size (float): Step size for Levy flight.
        Returns:
            list: Best nest found.
            float: Best cost.
    """
    best_cost = float('inf')
    best_nest = None
    top_nests = []  # Initialize an empty list for top nests

    for iteration in range(num_iterations):
        all_cost = []
        nests = custom_rng.initialize_nests(num_nests, num_vessels, a_range, b_range, c_min, c_max)

        for sublist in nests:
            individual_costs = cost_calculator.calculate_cost_component2(sublist)
            all_cost.append(individual_costs)

        min_cost = min(all_cost)
        best_nest_index = all_cost.index(min_cost)

        if 0 <= best_nest_index < len(nests):
            best_nest = nests[best_nest_index]

        num_abandon = int(0.2 * num_nests)
        low_fitness_nests = nests[:num_abandon]

        for nest_to_remove in low_fitness_nests:
            nests.remove(nest_to_remove)

        remaining_nests = nests
        updated_nests = []
        dimension = len(remaining_nests[0])

        for nest in remaining_nests:
            step_vector = LocalSearch.levy_flight_1(Lambda, dimension, step_size)
            updated_nest = [int(coord + step_size * step_item) for coord, step_item in zip(nest, step_vector)]
            updated_nests.append(updated_nest)
            
            update_cost = []
            for sublist_1 in updated_nests:
                updated_nest_cost = cost_calculator.calculate_cost_component2(sublist_1)
                update_cost.append(updated_nest_cost)

            sorted_nests = [nest for _, nest in sorted(zip(update_cost, nests))]  # Sort by cost

        # Calculate cosine similarity between initial nests and top nests
        initial_nests_array = np.array(nests)   
        top_nests_array = np.array(sorted_nests)
        similarities = cosine_similarity(initial_nests_array, top_nests_array)

        # Get the index of the most similar nest
        best_match_index = np.argmax(similarities)

        if 0 <= best_match_index < len(nests):
            if update_cost[0] < best_cost:
                best_cost = update_cost[0]
                best_nest = nests[best_match_index]

        # Add the top nests to the list
        top_nests = sorted_nests[:15]

    return top_nests, best_cost  # Return the list of top nests and best cost

# Example usage:
num_iterations = 30
Lambda = 1.5
step_size = 0.1

top_nests, best_cost = optimize_nests(num_iterations, num_nests, num_vessels, a_range, b_range, c_min, c_max, Lambda, step_size)

print("Top Nests (in ascending order of cost):")
for i, nest in enumerate(top_nests):
    print(f"Nest {i+1}: {nest}")
    print(f"Cost: {cost_calculator.calculate_cost_component2(nest)}")
print("Best Cost:")
print(best_cost)