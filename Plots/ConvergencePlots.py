import matplotlib.pyplot as plt

# Initialize lists to store best cost and iteration number
best_costs_over_time = []

def optimize_nests(num_iterations, num_nests, num_vessels, a_range, b_range, c_min, c_max,
                   Lambda, step_size):
    """
    Your existing optimization function with modifications to track best cost over time.
    """
    best_cost = float('inf')
    best_nest = None

    for iteration in range(num_iterations):
        # Existing code...

        # Check if the current nest's cost is better than the best cost found so far
        if update_cost[0] < best_cost:
            best_cost = update_cost[0]
            best_nests = top_nests

        # Store the best cost for each iteration
        best_costs_over_time.append(best_cost)

    return best_nests, best_cost

# Call the optimization function
top_nests, best_cost = optimize_nests(num_iterations, num_nests, num_vessels, a_range, b_range, c_min, c_max, Lambda, step_size)

# Plot the convergence curve
plt.figure()
plt.plot(range(1, num_iterations + 1), best_costs_over_time, marker='o', color='b', label='Best Cost')
plt.xlabel('Iteration')
plt.ylabel('Best Cost')
plt.title('Convergence Plot')
plt.grid(True)
plt.legend()
plt.show()