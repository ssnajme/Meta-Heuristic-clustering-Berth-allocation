import time, array, random, copy, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import algorithms, base, benchmarks, tools, creator
from deap import creator
from MainNestGeneration import custom_rng
from ObjectiveFunctions import cost_calculator
from MainNestGeneration import num_internal_arrays, num_elements, max_value


num_nests = int(input("Enter the number of vessels: "))
num_vessels = 1
a_range = 4
b_range = 6
c_min = 2
c_max = 4
top_nests_4 = []
top_costs_4 = []
# Function to pad nests to the maximum length  
def pad_nests(nests, max_length):  
    padded_nests = []  
    for nest in nests:  
        padded_nest = list(nest) + [0] * (max_length - len(nest))  # Padding with zeros  
        padded_nests.append(padded_nest)  
    return padded_nests  

# Particle class  
class Particle:  
    def __init__(self, position):  
        self.position = np.array(position, dtype=np.float64)  # Ensure position is of float type  
        self.velocity = np.random.uniform(-1, 1, size=self.position.shape).astype(np.float64)  # Ensure velocity is float  
        self.best_position = np.copy(self.position)  
        self.best_fitness = cost_calculator.calculate_cost_component2(self.position)  

nests = custom_rng.generate_sobol_integer_list(num_internal_arrays, num_elements, max_value)  

# Pad nests to the maximum length  
max_length = max(len(nest) for nest in nests)  
padded_nests = pad_nests(nests, max_length)  

num_iterations = 70  
num_particles = len(padded_nests)  
inertia_weight = 0.5  
cognitive_component = 1.0  
social_component = 1.0  

# Initialize particles based on padded nests  
particles = [Particle(np.array(nest, dtype=np.float64)) for nest in padded_nests]  # Ensure particle positions are float  
global_best_position = particles[0].best_position  
global_best_fitness = particles[0].best_fitness  

# Store global best positions for visualization  
best_positions = []  
convergence_costs =[]
# PSO Main Loop  
for iteration in range(num_iterations):  
    for particle in particles:  
        # Evaluate fitness  
        fitness = cost_calculator.calculate_cost_component2(particle.position)  

        # Update personal best  
        if fitness < particle.best_fitness:  
            particle.best_fitness = fitness  
            particle.best_position = np.copy(particle.position)  

        # Update global best  
        if fitness < global_best_fitness:  
            global_best_fitness = fitness  
            global_best_position = np.copy(particle.position) 
            convergence_costs.append(global_best_fitness)

            if len(top_nests_4) < 10:
                top_nests_4.append(particle.position)
                top_costs_4.append(fitness)
            else:
                worst_index = np.argmax(top_costs_4)
                top_nests_4[worst_index] = particle.position
                top_costs_4[worst_index] = fitness


    # Update particle velocities and positions  
    for particle in particles:  
        r1, r2 = np.random.rand(2)  
        cognitive_velocity = cognitive_component * r1 * (particle.best_position - particle.position)  
        social_velocity = social_component * r2 * (global_best_position - particle.position)  

        # Update velocity  
        particle.velocity = (inertia_weight * particle.velocity +  
                             cognitive_velocity + social_velocity)  

        # Update position  
        particle.position += particle.velocity  

    # Print the best fitness of the iteration
    # --- have this for cuckoo search as well ---  
    print(f"Iteration {iteration + 1}/{num_iterations}, Best Fitness: {global_best_fitness}")  

# Final best solution  
print("Final Global Best Position:", global_best_position)  
print("Final Global Best Fitness:", global_best_fitness)

print("Top 10 Nests:")
for i, nest in enumerate(top_nests_4):
    print(f"Nest {i + 1}: {nest}")
print("Top 10 Costs:")
list = []
for cost in top_costs_4:
    list.append(cost)
    print(cost)
    print(list)

# Visualization  

#--------------- thesis Table ---------------------------
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
num_iterations = 100
num_particles = 50

# Run the PSO algorithm
#global_best_position, global_best_fitness = pso_algorithm(num_iterations, num_particles)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Assuming you have the initial nests (initial_nests_array) and the top solution (global_best_position)
similarities = cosine_similarity(nests, global_best_position.reshape(1, -1))

# Calculate the average cosine similarity
uniqueness = 1 - similarities.mean()


# Calculate other metrics
all_nests_costs = [particle.best_fitness for particle in particles]  # Use all nests
avg_top_nests_cost = np.mean(all_nests_costs)
diversity = np.std(all_nests_costs)
num_successful_replacements = num_iterations  # All iterations involve updates

# Calculate uniqueness based on cosine similarity with all nests
similarities = cosine_similarity(global_best_position.reshape(1, -1), nests)
uniqueness = 1 - similarities.mean()

# You need to define the sample size for the confidence interval
sample_size = num_iterations
confidence_interval = 1.96 * (diversity / np.sqrt(sample_size))

# Print the adjusted results
print("Best Cost Found:", global_best_fitness)
print("Number of Iterations:", num_iterations)
print("Average Cost of Top Nests:", avg_top_nests_cost)
print("Diversity of Found Nests:", diversity)
print("Portion of Successfully Replaced Nests:", num_successful_replacements / num_iterations)
print("Uniqueness of Top Solution:", uniqueness)
print("Sample Size Confidence Interval:", confidence_interval)

#------------Thesis Table 2------- 
# 1) number of nests
# 2) number of iterations 
# 3) termination condition 
# 4) levy flight main parameter 
# 5) nest abandon rate 
# 6) objective function top cost
# 7) the running time 
# 8) convergence rate 
# 9) cognitive_velocity
# 10) cognitive_velocity
# 11) social_velocity 
# 12) End timing 

import numpy as np  
import time  
import psutil  
#from sklearn.metrics import cosine_similarity  

"""

# Take input for number of nests 
print("table 2 of thesis") 
num_nests = int(input("Enter the number of vessels: "))  
num_vessels = 1  
a_range = 4  
b_range = 6  
c_min = 2  
c_max = 4  
top_nests_4 = []  
top_costs_4 = []  

import numpy as np  
import time  
import psutil  # Make sure you have this installed with `pip install psutil`  

# Function to pad nests to the maximum length  
def pad_nests(nests, max_length):  
    padded_nests = []  
    for nest in nests:  
        padded_nest = list(nest) + [0] * (max_length - len(nest))  # Padding with zeros  
        padded_nests.append(padded_nest)  
    return padded_nests  

class Particle:  
    def __init__(self, position):  
        self.position = np.array(position, dtype=np.float64)  # Ensure position is of float type  
        self.velocity = np.random.uniform(-1, 1, size=self.position.shape).astype(np.float64)  # Velocity  
        self.best_position = np.copy(self.position)  
        self.best_fitness = cost_calculator.griewank_function(self.position)  

nests = custom_rng.initialize_nests(num_nests, num_vessels, a_range, b_range, c_min, c_max)  

# Pad nests to the maximum length  
max_length = max(len(nest) for nest in nests)  
padded_nests = pad_nests(nests, max_length)  

num_iterations = 70  
num_particles = len(padded_nests)  
inertia_weight = 0.5  
cognitive_component = 1.0  
social_component = 1.0   

# Initialize particles based on padded nests  
particles = [Particle(np.array(nest, dtype=np.float64)) for nest in padded_nests]  
global_best_position = particles[0].best_position  
global_best_fitness = particles[0].best_fitness  

# Store global best positions for visualization  
best_positions = []  
best_fitness_values = []  # To track the best fitness of each iteration  

top_nests_4 = []  
top_costs_4 = []  

# Start timing the optimization  
start_time = time.time()  

# PSO Main Loop  
for iteration in range(num_iterations):  
    for particle in particles:  
        # Evaluate fitness  
        fitness = cost_calculator.griewank_function(particle.position)  

        # Update personal best  
        if fitness < particle.best_fitness:  
            particle.best_fitness = fitness  
            particle.best_position = np.copy(particle.position)  

        # Update global best  
        if fitness < global_best_fitness:  
            global_best_fitness = fitness  
            global_best_position = np.copy(particle.position)   

            if len(top_nests_4) < 10:  
                top_nests_4.append(particle.position)  
                top_costs_4.append(fitness)  
            else:  
                worst_index = np.argmax(top_costs_4)  
                top_nests_4[worst_index] = particle.position  
                top_costs_4[worst_index] = fitness  

    # Update particle velocities and positions  
    for particle in particles:  
        r1, r2 = np.random.rand(2)  
        cognitive_velocity = cognitive_component * r1 * (particle.best_position - particle.position)  
        social_velocity = social_component * r2 * (global_best_position - particle.position)  

        # Update velocity  
        particle.velocity = (inertia_weight * particle.velocity +  
                             cognitive_velocity + social_velocity)  

        # Update position  
        particle.position += particle.velocity  

    # Store the best fitness of the iteration  
    best_fitness_values.append(global_best_fitness)  
    print(f"Iteration {iteration + 1}/{num_iterations}, Best Fitness: {global_best_fitness}")  

# End timing  
end_time = time.time()  
running_time = end_time - start_time  

# Calculate convergence rate (average improvement per generation)  
if len(best_fitness_values) > 1:  
    convergence_rate = np.mean(np.diff(best_fitness_values))  
else:  
    convergence_rate = 0  # Not enough data to calculate improvement  

# Get CPU and RAM usage  
cpu_usage = psutil.cpu_percent(interval=None)  
memory_usage = psutil.virtual_memory().percent  

# Print final results  
print("Final Global Best Position:", global_best_position)  
print("Final Global Best Fitness:", global_best_fitness)  

print("Top 10 Nests:")  
for i, nest in enumerate(top_nests_4):  
    print(f"Nest {i + 1}: {nest}")  
print("Top 10 Costs:")  
for cost in top_costs_4:  
    print(cost)  

# Print the requested metrics  
print("1) Number of nests:", num_nests)  
print("2) Number of iterations:", num_iterations)  
print("3) Termination condition: Maximum iterations reached")  
print("4) Levy flight main parameter: N/A for PSO")  # As we don't use Levy flight in PSO  
print("5) Nest abandon rate: N/A for PSO")  # Nest abandonment is not present in PSO  
print("6) Objective function top cost:", global_best_fitness)  
print("7) The running time (in seconds):", running_time)  
print("8) Convergence rate (Average Improvement per Iteration):", convergence_rate)  
print("9) CPU Usage (%):", cpu_usage)  
print("10) Memory Usage (%):", memory_usage)  

# To indicate when the algorithm finished  
print("12) End timing:", end_time)
""" 

import numpy as np  
import matplotlib.pyplot as plt  


# Objective function to minimize with randomness  
def objective_function(x):  
    # Adding randomness to simulate real-world scenarios  
    noise = np.random.uniform(0, 1)  # Random noise between 0 and 1  
    cost = np.sum(np.square(x)) + noise + 2  # Ensure the cost is always above 2  
    return cost  

# Particle class representing each particle in the swarm  
class Particle:  
    def __init__(self, dimensions):  
        self.position = np.random.rand(dimensions) * 10 - 5  # Initialize in range [-5, 5]  
        self.velocity = np.random.rand(dimensions) * 0.1 - 0.05  # Initialize small velocity  
        self.best_position = np.copy(self.position)  
        self.best_cost = objective_function(self.position)  

# PSO Class managing the swarm  
class PSO:  
    def __init__(self, num_particles, dimensions, num_iterations):  
        self.num_particles = num_particles  
        self.dimensions = dimensions  
        self.num_iterations = num_iterations  
        self.particles = [Particle(dimensions) for _ in range(num_particles)]  
        self.global_best_position = np.copy(self.particles[0].best_position)  
        self.global_best_cost = self.particles[0].best_cost  
        self.best_costs = []  

    def update_particles(self):  
        for particle in self.particles:  
            # Update velocity  
            inertia_weight = 0.5  
            cognitive_weight = 1.5  
            social_weight = 1.5  

            r1, r2 = np.random.rand(2)  
            particle.velocity = (inertia_weight * particle.velocity +  
                                 cognitive_weight * r1 * (particle.best_position - particle.position) +  
                                 social_weight * r2 * (self.global_best_position - particle.position))  

            # Update position  
            particle.position += particle.velocity  
            
            # Check boundaries (simple clamping)  
            particle.position = np.clip(particle.position, -5, 5)  

            # Evaluate cost with randomness  
            cost = objective_function(particle.position)  

            # Update personal best  
            if cost < particle.best_cost:  
                particle.best_cost = cost  
                particle.best_position = np.copy(particle.position)  

            # Update global best  
            if cost < self.global_best_cost:  
                self.global_best_cost = cost  
                self.global_best_position = np.copy(particle.position)  

    def run(self):  
        for iteration in range(self.num_iterations):  
            self.update_particles()  
            # Store the global best cost for plotting later  
            self.best_costs.append(self.global_best_cost)  
            print(f"Iteration {iteration + 1}/{self.num_iterations}, Best Cost: {self.global_best_cost}")  

        return self.global_best_cost, self.global_best_position  

# Parameters  
num_particles = 30  
dimensions = 2  
num_iterations = 50  

# Run PSO  
pso = PSO(num_particles, dimensions, num_iterations)  
best_cost, best_position = pso.run()  

# Plotting the convergence cost  
plt.figure(figsize=(10, 5))  
plt.plot(range(1, num_iterations + 1), pso.best_costs, marker='o', color='blue')  
plt.title('PSO Convergence Plot with Randomized Costs')  
plt.xlabel('Iterations')  
plt.ylabel('Best Cost (Objective Function Value)')  
plt.grid(True)  
plt.ylim(2, np.max(pso.best_costs) + 1)  # Set y-axis starting from 2 to show the range  
plt.axhline(y=2, color='r', linestyle='--', label='Cost = 2')  
plt.legend()  
plt.show()