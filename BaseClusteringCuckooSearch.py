import numpy as np
import warnings
import psutil
import time
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from LocalSearches import LocalSearch
from ObjectiveFunctions import cost_calculator
from MainNestGeneration import num_internal_arrays, num_elements, max_value

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


best_cost = float("inf")
best_nest = None

# Start measuring time
start_time = time.time()


def plot_convergence(convergence_costs):
    # Plot the convergence over iterations
    plt.plot(range(1, len(convergence_costs) + 1), convergence_costs)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.title("Convergence Plot")
    plt.grid(True)
    plt.show()


def optimize_nests(
    num_iterations,
    num_nests,
    num_vessels,
    a_range,
    b_range,
    c_min,
    c_max,
    Lambda,
    step_size,
):
    best_cost = float("inf")
    best_nest = None
    top_nests = []  # Initialize an empty list for top nests
    convergence_costs = []
    cpu_usages = []
    memory_usages = []
    for iteration in range(num_iterations):
        all_cost = []
        # Initialize lists to store convergence data
        convergence_costs = []
        #nests = custom_rng.initialize_nests(
        #    num_nests, num_vessels, a_range, b_range, c_min, c_max
        #)

        nests = custom_rng.generate_sobol_integer_list(num_internal_arrays, num_elements, max_value)  

        if iteration % 10 == 0:  # Record every 10 iterations
            memory_usage = psutil.virtual_memory().percent
            memory_usages.append(memory_usage)

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
            updated_nest = [
                int(coord + step_size * step_item)
                for coord, step_item in zip(nest, step_vector)
            ]
            updated_nests.append(updated_nest)

            update_cost = []
            for sublist_1 in updated_nests:
                updated_nest_cost = cost_calculator.calculate_cost_component2(sublist_1)
                update_cost.append(updated_nest_cost)

            sorted_nests = [
                nest for _, nest in sorted(zip(update_cost, nests))
            ]  # Sort by cost

        # Apply Gaussian Mixture clustering to the nests
        gmm = GaussianMixture(n_components=15, random_state=42)
        gmm.fit(sorted_nests)

        top_nests = gmm.means_
        best_match_index = gmm.predict([best_nest])[0]

        if update_cost[0] < best_cost:
            best_cost = update_cost[0]
            best_nest = nests[best_match_index]

            # Get the CPU usage
        cpu_usage = psutil.cpu_percent(interval=None)  # Get CPU usage as a percentage
        cpu_usages.append(cpu_usage)  # change this ------

        # Collecting data for every 10th iteration
        if iteration % 10 == 0:
            print(f"\nIteration {iteration}:")
            print(
                f"Top Nest Costs: {[cost_calculator.calculate_cost_component2(nest) for nest in top_nests]}"
            )
            print(f"CPU Usage: {cpu_usage}%")
            print(f"RAM Usage: {memory_usage}%")

    for nest in top_nests:
        nest_cost = cost_calculator.calculate_cost_component2(nest)
        convergence_costs.append(nest_cost)

    # Calculate running time
    end_time = time.time()
    running_time = end_time - start_time

    # Calculate convergence rate as the percentage of the least cost over the mean of convergence costs
    convergence_rate = (
        min(convergence_costs) / (sum(convergence_costs) / len(convergence_costs))
        if convergence_costs
        else None
    )

    return (
        top_nests,
        best_cost,
        convergence_costs,
        cpu_usages,
        memory_usages,
        convergence_rate,
        running_time,
    )  # Return the list of top nests and best cost


# Example usage:
num_iterations = 30
Lambda = 1.5
step_size = 0.1

top_nests_5, best_cost_5, convergence_costs, cpu_usages, memory_usages, convergence_rate, running_time = optimize_nests(
    num_iterations,
    num_nests,
    num_vessels,
    a_range,
    b_range,
    c_min,
    c_max,
    Lambda,
    step_size,
)

print(convergence_costs)
convergence_costs_sorted = sorted(convergence_costs, reverse=True)
plot_convergence(convergence_costs_sorted)

# --- other 4 plots here -----
# --- CPU per iteration plots ---
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)  # Subplot for CPU Usage
plt.plot(cpu_usages, marker="o", color="blue")
plt.title("CPU Usage over Iterations")
plt.xlabel("Iteration")
plt.ylabel("CPU Usage (%)")
plt.grid(True)
plt.xticks(np.arange(0, num_iterations + 1, step=10))
plt.yticks(np.arange(0, 101, step=10))
plt.ylim(0, 100)
plt.xlim(0, num_iterations)

# Plot Memory Usage
plt.subplot(2, 1, 2)  # Subplot for Memory Usage
plt.plot(memory_usages, marker="o", color="green")
plt.title("Memory Usage over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Memory Usage (%)")
plt.grid(True)
plt.xticks(np.arange(0, num_iterations + 1, step=10))
plt.yticks(np.arange(0, 101, step=10))
plt.ylim(0, 100)
plt.xlim(0, num_iterations)

plt.tight_layout()
plt.show()

# ---- sensitivity analysis ----


# ---- Robustness analysis -----


# --------------- thesis Table ---------------------------
"""
1) the best cost found by the algorithm 
2) the number of iterations of the algorithm 
3) the average cost of top found nests
4) the diversity of the found nests
5) the portion of the nests that have been successfully replaced 
6) measuring the uniqueness of the top solution 
7) the total convergence time of the algorithm 
8) the sample size confidence interval



print("############")
print("thesis table")
from sklearn.metrics.pairwise import cosine_similarity


def optimize_nests(
    num_iterations,
    num_nests,
    num_vessels,
    a_range,
    b_range,
    c_min,
    c_max,
    Lambda,
    step_size,
):
    best_cost = float("inf")
    best_nest = None
    top_nests = []  # Initialize an empty list for top nests
    convergence_costs = []
    cpu_usages = []
    memory_usages = []
    for iteration in range(num_iterations):
        all_cost = []
        # Initialize lists to store convergence data
        convergence_costs = []
        nests = custom_rng.initialize_nests(
            num_nests, num_vessels, a_range, b_range, c_min, c_max
        )

        if iteration % 10 == 0:  # Record every 10 iterations
            memory_usage = psutil.virtual_memory().percent
            memory_usages.append(memory_usage)

        for sublist in nests:
            individual_costs = cost_calculator.griewank_function(sublist)
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
            updated_nest = [
                int(coord + step_size * step_item)
                for coord, step_item in zip(nest, step_vector)
            ]
            updated_nests.append(updated_nest)

            update_cost = []
            for sublist_1 in updated_nests:
                updated_nest_cost = cost_calculator.griewank_function(sublist_1)
                update_cost.append(updated_nest_cost)

            sorted_nests = [
                nest for _, nest in sorted(zip(update_cost, nests))
            ]  # Sort by cost

        # Apply Gaussian Mixture clustering to the nests
        gmm = GaussianMixture(n_components=15, random_state=42)
        gmm.fit(sorted_nests)

        top_nests = gmm.means_
        best_match_index = gmm.predict([best_nest])[0]

        if update_cost[0] < best_cost:
            best_cost = update_cost[0]
            best_nest = nests[best_match_index]

    for nest in top_nests:
        nest_cost = cost_calculator.griewank_function(nest)
        convergence_costs.append(nest_cost)

        # Get the CPU usage
        cpu_usage = psutil.cpu_percent(interval=None)  # Get CPU usage as a percentage
        cpu_usages.append(cpu_usage)  # change this ------


    similarities = cosine_similarity(nests, top_nests)

    return (
        top_nests,
        best_cost,
        convergence_costs,
        cpu_usages,
        memory_usages, 
        similarities 
    )  # Return the list of top nests and best cost


# Example usage:
num_iterations = 30
Lambda = 1.5
step_size = 0.1

top_nests,best_cost, convergence_costs, cpu_usages, memory_usages, similarities = optimize_nests(
    num_iterations,
    num_nests,
    num_vessels,
    a_range,
    b_range,
    c_min,
    c_max,
    Lambda,
    step_size,
)

print("thesis table")
# Calculate the average cosine similarity
uniqueness = 1 - similarities.mean()


# Calculate the requested metrics
num_iterations = num_iterations
avg_top_nests_cost = sum(convergence_costs) / len(convergence_costs)
diversity = np.std(convergence_costs)
num_successful_replacements = len(convergence_costs) - 1  # Subtract 1 for the initial best cost
uniqueness = 1 - similarities.mean()  # You need to define `similarities`
# You can add timing measurements for convergence time if needed

# Calculate sample size confidence interval (you need to define the sample size)
sample_size = len(convergence_costs)  # Define your actual sample size
confidence_interval = 1.96 * (diversity / np.sqrt(sample_size))

# Print the results
print("Best Cost:", best_cost)
print("Number of Iterations:", num_iterations)
print("Average Cost of Top Nests:", avg_top_nests_cost)
print("Diversity of Found Nests:", diversity)
print("Portion of Successfully Replaced Nests:", num_successful_replacements / num_iterations)
print("Uniqueness of Top Solution:", uniqueness)
print("Sample Size Confidence Interval:", confidence_interval)
"""

# ------------Thesis Table 2-------
# 1) number of nests
# 2) number of iterations
# 3) termination condition
# 4) levy flight main parameter
# 5) nest abandon rate
# 6) objective function top cost
# 7) the running time
# 8) convergence rate
# Measure the elapsed time
# Parameters
num_iterations = 30
num_nests = 100  # Example value, set according to your context
num_vessels = 10  # Example value
a_range = (0, 100)  # Example bounds
b_range = (0, 100)  # Example bounds
c_min = 0  # Example minimum value
c_max = 100  # Example maximum value
Lambda = 1.5
step_size = 0.1
nest_abandon_rate = 0.2  # 20% of nests will be abandoned

end_time = time.time()
running_time = end_time - start_time

# Calculate convergence rate as the percentage of the least cost over the mean of convergence costs
if convergence_costs:  # Check if we have any convergence costs recorded
    convergence_rate = min(convergence_costs) / (
        sum(convergence_costs) / len(convergence_costs)
    )
else:
    convergence_rate = None  # or 0, based on your requirement

# Print results
print("1) Number of nests:", num_nests)
print("2) Number of iterations:", num_iterations)
print(
    "3) Termination condition: Using maximum iterations"
)  # Update this if you have a different condition
print("4) Levy flight main parameter (Lambda):", Lambda)
print("5) Nest abandon rate:", nest_abandon_rate)
print("6) Objective function top cost:", best_cost)
print("7) The running time (in seconds):", running_time)
print("8) Convergence rate:", convergence_rate)
