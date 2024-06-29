import numpy as np
import csv
import random
import warnings
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from LocalSearches import LocalSearch
from ObjectiveFunctions import cost_calculator
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


num_iterations = 300
weights = [1.1, 1.2, 1.3, 1.5]
penalty_weight = 1.0
T_start = 0
T_end = 100

Lambda = 1.5
dimension = 3
step_size = 0.1


best_cost = float('inf')
best_nest = None


def plot_convergence(convergence_costs):
    # Plot the convergence over iterations
    plt.plot(range(1, len(convergence_costs) + 1), convergence_costs)
    plt.xlabel('Iteration')
    plt.ylabel('Best Cost')
    plt.title('Convergence Plot')
    plt.grid(True)
    plt.show()

def optimize_nests(num_iterations, num_nests, num_vessels, a_range, b_range, c_min, c_max, Lambda, step_size):
    best_cost = float('inf')
    best_nest = None
    top_nests = []  # Initialize an empty list for top nests

    for iteration in range(num_iterations):
        all_cost = []
        # Initialize lists to store convergence data
        convergence_costs = []
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

        # Apply K-means clustering to the nests
        kmeans = KMeans(n_clusters=15, n_init=10, random_state=42)
        kmeans.fit(sorted_nests)
        top_nests = kmeans.cluster_centers_

        # Get the index of the most similar nest
        best_match_index = kmeans.predict([best_nest])[0]

        if update_cost[0] < best_cost:

            best_cost = update_cost[0]
            best_nest = nests[best_match_index]



    for nest in top_nests:
        nest_cost = cost_calculator.calculate_cost_component2(nest)
        convergence_costs.append(nest_cost)
   
        #print(convergence_costs)


    return top_nests, best_cost, convergence_costs  # Return the list of top nests and best cost

# Example usage:
num_iterations = 30
Lambda = 1.5
step_size = 0.1

top_nests, best_cost, convergence_costs = optimize_nests(num_iterations, num_nests, num_vessels, a_range, b_range, c_min, c_max, Lambda, step_size)

print(convergence_costs)
convergence_costs_sorted = sorted(convergence_costs, reverse=True)
plot_convergence(convergence_costs_sorted)


# Create a convergence plot in descending order using Plotly
fig = go.Figure(data=go.Scatter(x=list(range(len(convergence_costs_sorted))), y=convergence_costs_sorted, mode='lines+markers'))
fig.update_layout(title='Convergence Plot (Descending Order)', xaxis_title='Iteration', yaxis_title='Cost')
fig.show()


# bar plot 
fig = go.Figure(data=go.Bar(x=list(range(len(convergence_costs))), y=convergence_costs))
fig.update_layout(title='Convergence Costs over Iterations', xaxis_title='Iteration', yaxis_title='Cost')
fig.show()

plt.figure()
plt.bar(range(len(convergence_costs)), convergence_costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Convergence Costs over Iterations')
plt.show()

