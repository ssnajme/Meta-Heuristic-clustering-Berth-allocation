import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import plotly.graph_objects as go
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
from MainNestGeneration import num_internal_arrays, num_elements, max_value

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


# ----- comment for the thesis table -------------


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
    cost_function,
):
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
    best_cost = float("inf")
    best_nest = None
    top_nests = []  # Initialize an empty list for top nests
    convergence_costs = []
    cpu_usages = []
    memory_usages = []
    for iteration in range(num_iterations):
        all_cost = []
        nests = custom_rng.initialize_nests(
          num_nests, num_vessels, a_range, b_range, c_min, c_max
        )
        
        #nests = custom_rng.generate_sobol_integer_list(num_internal_arrays, num_elements, max_value)  

        if iteration % 10 == 0:  # Record every 10 iterations
            memory_usage = psutil.virtual_memory().percent
            memory_usages.append(memory_usage)

        for sublist in nests:
            individual_costs = cost_function(sublist)
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
            # updated_nest = [coord + step_size * step_item for coord, step_item in zip(nests[i], step_vector)]
            updated_nests.append(updated_nest)

            update_cost = []
            for sublist_1 in updated_nests:
                updated_nest_cost = cost_function(sublist_1)
                update_cost.append(updated_nest_cost)

            sorted_nests = [
                nest for _, nest in sorted(zip(update_cost, nests))
            ]  # Sort by cost

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
        top_nests = sorted_nests[:10]

        # --- need to return the convergence costs here ---
        for nest in top_nests:
            nest_cost = cost_function(nest)
            convergence_costs.append(nest_cost)

            # Get the CPU usage
            cpu_usage = psutil.cpu_percent(
                interval=None
            )  # Get CPU usage as a percentage
            cpu_usages.append(cpu_usage)  # change this ------

    return (
        top_nests,
        best_cost,
        convergence_costs,
        cpu_usages,
        memory_usages,
    )  # Return the list of top nests and best cost


"""
# Example usage:
num_iterations = 80
Lambda = 2
step_size = 1.5

# --- This is for running the above algorithm ---

all_best_costs = []
all_convergence_costs = []
all_cpu_usages = []
num_runs = 35
for run in range(num_runs):
    top_nests, best_cost, convergence_costs, cpu_usages,  memory_usages = optimize_nests(
        num_iterations,
        num_nests,
        num_vessels,
        a_range,
        b_range,
        c_min,
        c_max,
        Lambda,
        step_size,
        cost_function=cost_calculator.calculate_cost_component2,
    )

    all_best_costs.append(best_cost)
    all_convergence_costs.extend(convergence_costs)
    all_cpu_usages.extend(cpu_usages)

    print(f"Run {run + 1}/{num_runs}: Best Cost = {best_cost}")
"""
##########################################################num_iterations, num_nests, num_vessels, a_range, b_range, c_min, c_max, Lambda, step_size, cost_function

top_nests, best_cost, convergence_costs, cpu_usages, memory_usages = optimize_nests(
    num_iterations,
    num_nests,
    num_vessels,
    a_range,
    b_range,
    c_min,
    c_max,
    Lambda,
    step_size,
    cost_function=cost_calculator.calculate_cost_component2,
)
print("convergence_costs")
print(convergence_costs)
print("Top Nests (in ascending order of cost):")
cost_values = []
for i, nest in enumerate(top_nests):
    print(f"Nest {i+1}: {nest}")
    print(f"Cost: {cost_calculator.calculate_cost_component2(nest)}")
    cost_values.append(cost_calculator.calculate_cost_component2(nest))


# ----- Here a convergence plot with sns -------------


# -- here len(convergence_costs) is weirdly long ---
def plot_convergence(convergence_costs):
    sorted_costs = sorted(convergence_costs, reverse=True)  # Sort costs in descending order
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(sorted_costs) + 1), sorted_costs, marker='o', linestyle='-', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Convergence Plot of the Base cuckoo search')
    plt.grid(True)
    plt.show()


convergence_costs_sorted = sorted(convergence_costs, reverse=True)
plot_convergence(convergence_costs_sorted[0:57])
""" 
# ----- CPU per iterations plots ----------
# ----- a single call to the function ----

top_nests, best_cost, convergence_costs, cpu_usages,  memory_usages = optimize_nests(  
    num_iterations,  
    num_nests,  
    num_vessels,  
    a_range,  
    b_range,  
    c_min,  
    c_max,  
    Lambda,  
    step_size,  
    cost_function=cost_calculator.calculate_cost_component2  
)  

# Plotting CPU Usage  
# Plot CPU Usage  
# ----- Memory per iteration plots --------
# ----- For the base cuckoo search only ---

plt.figure(figsize=(10, 5))  
plt.subplot(2, 1, 1)  # Subplot for CPU Usage  
plt.plot(cpu_usages, marker='o', color='blue')  
plt.title('CPU Usage over Iterations')  
plt.xlabel('Iteration')  
plt.ylabel('CPU Usage (%)')  
plt.grid(True)  
plt.xticks(np.arange(0, num_iterations + 1, step=10))  
plt.yticks(np.arange(0, 101, step=10))  
plt.ylim(0, 100)  
plt.xlim(0, num_iterations)  

# Plot Memory Usage  
plt.subplot(2, 1, 2)  # Subplot for Memory Usage  
plt.plot(memory_usages, marker='o', color='green')  
plt.title('Memory Usage over Iterations')  
plt.xlabel('Iteration')  
plt.ylabel('Memory Usage (%)')  
plt.grid(True)  
plt.xticks(np.arange(0, num_iterations + 1, step=10))  
plt.yticks(np.arange(0, 101, step=10))  
plt.ylim(0, 100)  
plt.xlim(0, num_iterations)  

# Adjust layout to prevent overlap  
plt.tight_layout()  
plt.show()  


# ---- Algorithm Sensitivity Analysis  ---- General ----

num_iterations_values = [30, 50, 100, 150, 200]
Lambda_values = [1.5, 1.78, 2.0]
step_size_values = [1.1, 2.5, 2.8]
cost_values = []
results = []

# Perform sensitivity analysis   

for num_iterations in num_iterations_values:  
    for Lambda in Lambda_values:  
        for step_size in step_size_values:  
            best_nest, best_cost, convergence_costs,cpu_usages,memory_usages = optimize_nests(num_iterations, num_nests, num_vessels, a_range, b_range, c_min, c_max, Lambda, step_size, cost_calculator.calculate_cost_component2)  
            
            # Clear cost_values before appending new values for each iteration  
            cost_values.clear()  
            for nest in best_nest:  # Assuming best_nest is a list of nests  
                cost_values.append(cost_calculator.calculate_cost_component2(nest))  
                
            # Collecting the results for each cost value  
            for cost_value in cost_values:  
                results.append({  
                    'num_iterations': num_iterations,   
                    'Lambda': Lambda,   
                    'step_size': step_size,   
                    'cost': cost_value  
                })

# Assuming `results` is your list of dictionaries created previously  
# Step 1: Convert the results into a DataFrame  
df_results = pd.DataFrame(results)  

# Step 2: Create the line plot  
plt.figure(figsize=(12, 6))  # Optional: Adjust the figure size  
sns.lineplot(data=df_results, x='num_iterations', y='cost', hue='Lambda', style='step_size', markers=True)  

# Step 3: Add titles and labels  
plt.title('Cost Analysis for Different Parameters')  
plt.xlabel('Number of Iterations')  
plt.ylabel('Cost')  
plt.legend(title='Lambda and Step Size')  
plt.grid(True)  

# Step 4: Show the plot  
plt.show() 


###########-Evaluate the robustness-#############
"""
"""
function_map = {
    "calculate_cost_component2": cost_calculator.calculate_cost_component2,
    "rosenbrock": cost_calculator.rosenbrock_function,
    "ackley": cost_calculator.ackley_function, 
    "rastrigin": cost_calculator.rastrigin_function,
    "griewank": cost_calculator.griewank_function,
    "schwefel": cost_calculator.schwefel_function,
}

# List of function names, using the keys from the function_map
function_names = list(function_map.keys())

# Using a dictionary comprehension to call the `optimize_nests` method with the actual functions
results = {
    function_name: optimize_nests(
        num_iterations,
        num_nests,
        num_vessels,
        a_range,
        b_range,
        c_min,
        c_max,
        Lambda,
        step_size,
        function_map[function_name],  # Use the mapping here
    )
    for function_name in function_names
}
print(results)
# ---- Visualization**: Create additional plots (e.g., box plots, heatmaps) to visualize the sensitivity of algorithms to different parameters.

# Perform one-way ANOVA on best costs  
cost_values = {key: value[1] for key, value in results.items()} 
calculate_cost_component2 = [np.mean(run) for run in results['calculate_cost_component2'][0]]
rosenbrock = [np.mean(run) for run in results['rosenbrock'][0]]
rastrigin = [np.mean(run) for run in results['rastrigin'][0]]
griewank = [np.mean(run) for run in results['griewank'][0]]
schwefel = [np.mean(run) for run in results['schwefel'][0]]

# Perform ANOVA
anova_results = stats.f_oneway(calculate_cost_component2, rosenbrock, rastrigin, griewank, schwefel)
print(f"ANOVA results: F-value = {anova_results.statistic}, p-value = {anova_results.pvalue}")

#--- Prepare data for plotting --------------
# -- create a box and whisker plot for this -

#Prepare data for ANOVA  
anova_results = stats.f_oneway(  
    calculate_cost_component2,   
    rosenbrock,   
    rastrigin,   
    griewank,   
    schwefel  
)  

# Prepare data for box plot  
data = [  
    calculate_cost_component2,   
    rosenbrock,   
    rastrigin,   
    griewank,   
    schwefel  
]  

# Create box and whisker plot  
plt.figure(figsize=(10, 6))  
plt.boxplot(data, labels=[  
    'Calculate Cost Component 2',   
    'Rosenbrock',   
    'Rastrigin',   
    'Griewank',   
    'Schwefel'  
])  
plt.title('Box and Whisker Plot of Cost Values')  
plt.ylabel('Mean Cost Value')  
plt.grid(axis='y')    
plt.tight_layout()  
plt.show() 
"""
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


"""


# ------------- thesis table code --------------------------
print("table thesis")
print("######")


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
    cost_function,
):
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
    best_cost = float("inf")
    best_nest = None
    top_nests = []  # Initialize an empty list for top nests
    convergence_costs = []
    cpu_usages = []
    memory_usages = []
    for iteration in range(num_iterations):
        all_cost = []
        #nests = custom_rng.initialize_nests(
        #    num_nests, num_vessels, a_range, b_range, c_min, c_max
        #)
        nests = custom_rng.generate_sobol_integer_list(num_internal_arrays, num_elements, max_value)  
        if iteration % 10 == 0:  # Record every 10 iterations
            memory_usage = psutil.virtual_memory().percent
            memory_usages.append(memory_usage)

        for sublist in nests:
            individual_costs = cost_function(sublist)
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
            # updated_nest = [coord + step_size * step_item for coord, step_item in zip(nests[i], step_vector)]
            updated_nests.append(updated_nest)

            update_cost = []
            for sublist_1 in updated_nests:
                updated_nest_cost = cost_function(sublist_1)
                update_cost.append(updated_nest_cost)

            sorted_nests = [
                nest for _, nest in sorted(zip(update_cost, nests))
            ]  # Sort by cost

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
        top_nests = sorted_nests[:10]

        # --- need to return the convergence costs here ---
        for nest in top_nests:
            nest_cost = cost_function(nest)
            convergence_costs.append(nest_cost)

            # Get the CPU usage
            cpu_usage = psutil.cpu_percent(
                interval=None
            )  # Get CPU usage as a percentage
            cpu_usages.append(cpu_usage)  # change this ------

    # Calculate the requested metrics
    best_cost = min(all_cost)
    num_successful_replacements = len(low_fitness_nests)
    top_nests_costs = [cost_function(nest) for nest in sorted_nests]
    avg_top_nests_cost = sum(top_nests_costs) / len(top_nests_costs)
    diversity = np.std(top_nests_costs)
    uniqueness = 1 - similarities.mean()

    # You can add timing measurements for convergence time if needed

    # Calculate sample size confidence interval (you need to define the sample size)
    sample_size = len(nests)  # Define your actual sample size
    confidence_interval = 1.96 * (diversity / np.sqrt(sample_size))

    return (
        best_cost,
        num_iterations,
        avg_top_nests_cost,
        diversity,
        num_successful_replacements,
        uniqueness,
        confidence_interval,
    )  # Return the list of top nests and best cost


# Extract the individual values from the result tuple
(
    best_cost,
    num_iterations,
    avg_top_nests_cost,
    diversity,
    num_successful_replacements,
    uniqueness,
    confidence_interval,
) = optimize_nests(
    num_iterations,
    num_nests,
    num_vessels,
    a_range,
    b_range,
    c_min,
    c_max,
    Lambda,
    step_size,
    cost_function=cost_calculator.calculate_cost_component2,
)

# print("Best Cost:", best_cost)
# print("Number of Iterations:", num_iterations)
# print("Average Cost of Top Nests:", avg_top_nests_cost)
# print("Diversity of Found Nests:", diversity)
# print(
#    "Portion of Successfully Replaced Nests:", num_successful_replacements / num_nests
# )
# print("Uniqueness of Top Solution:", uniqueness)
# print("Sample Size Confidence Interval:", confidence_interval)


# ------------Thesis Table 2-------
# 1) number of nests
# 2) number of iterations
# 3) termination condition
# 4) levy flight main parameter
# 5) nest abandon rate
# 6) objective function top cost
# 7) the running time
# 8) convergence rate
# 9) End timing
# 10) cpu usage
# 11) ram usage
# -----------------------------------
import psutil
import numpy as np
import time
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
    cost_function,
):
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
    start_time = time.time()  # Start timing
    best_cost = float("inf")
    best_nest = None
    top_nests = []  # Initialize an empty list for top nests
    convergence_costs = []
    cpu_usages = []
    memory_usages = []

    # Print initialization details
    print("Optimization Parameters:")
    print(f"1) Number of nests: {num_nests}")
    print(f"2) Number of iterations: {num_iterations}")
    print(f"3) Termination condition: {num_iterations} iterations")
    print(f"4) Levy flight main parameter (Lambda): {Lambda}")
    print(f"5) Nest abandon rate: 20%")

    for iteration in range(num_iterations):
        all_cost = []
        #nests = custom_rng.initialize_nests(
        #    num_nests, num_vessels, a_range, b_range, c_min, c_max
        #)
        nests = custom_rng.generate_sobol_integer_list(num_internal_arrays, num_elements, max_value)  

        if iteration % 10 == 0:  # Record every 10 iterations
            memory_usage = psutil.virtual_memory().percent
            memory_usages.append(memory_usage)

        for sublist in nests:
            individual_costs = cost_function(sublist)
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
            updated_nest_cost = cost_function(sublist_1)
            update_cost.append(updated_nest_cost)

        sorted_nests = [
            nest for _, nest in sorted(zip(update_cost, nests))
        ]  # Sort by cost

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
        top_nests = sorted_nests[:10]

        # --- need to return the convergence costs here ---
        for nest in top_nests:
            nest_cost = cost_function(nest)
            convergence_costs.append(nest_cost)

            # Get the CPU usage
            cpu_usage = psutil.cpu_percent(
                interval=None
            )  # Get CPU usage as a percentage
            cpu_usages.append(cpu_usage)

        # Print the top cost and CPU/RAM usage at the end of each iteration (optional)
        if iteration % 10 == 0:  # Print every 10 iterations
            print(f"\nIteration: {iteration}")
            print(f"Objective function top cost: {min_cost}")
            print(f"CPU Usage: {cpu_usage}%")
            print(f"RAM Usage: {memory_usage}%")

    # Collect metrics after the iterations are done
    end_time = time.time()  # End timing
    running_time = end_time - start_time
    convergence_rate = len(convergence_costs) / num_iterations  # Example calculation
    print("\nOptimization Completed.")
    print(f"6) Objective function top cost: {best_cost}")
    print(f"7) The running time: {running_time:.2f} seconds")
    print(f"8) Convergence rate: {convergence_rate:.2f}")
    print(f"9) End timing: {end_time:.2f} seconds")
    print(f"10) Average CPU Usage: {np.mean(cpu_usages):.2f}%")
    print(f"11) Average RAM Usage: {np.mean(memory_usages):.2f}%")

    return (
        top_nests,
        best_cost,
        convergence_costs,
        cpu_usages,
        memory_usages,
    )  # Return the list of top nests and best cost


# Call the optimize_nests function with the defined parameters
top_nests, best_cost, convergence_costs, cpu_usages, memory_usages = optimize_nests(
    num_iterations,
    num_nests,
    num_vessels,
    a_range,
    b_range,
    c_min,
    c_max,
    Lambda,
    step_size,
    cost_function=cost_calculator.calculate_cost_component2,
)

# Optionally, print the results after optimization
print("\nFinal Results:")
print("Top Nests:", top_nests)
print("Best Cost Achieved:", best_cost)
print("Convergence Costs:", convergence_costs)
print("CPU Usages:", cpu_usages)
print("Memory Usages:", memory_usages)
print(type(nests))