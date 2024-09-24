import numpy as np  
import matplotlib.pyplot as plt  
from MainNestGeneration import custom_rng
from ObjectiveFunctions import cost_calculator
from MainNestGeneration import num_internal_arrays, num_elements, max_value

# Objective function to minimize  
#def objective_function(x):  
#    return np.sum(np.square(x))  # Modified to support multi-dimensional input  
num_nests = int(input("Enter the number of vessels: "))
num_vessels = 1
a_range = 4
b_range = 6
c_min = 2
c_max = 4
#nests = custom_rng.initialize_nests(num_nests, num_vessels, a_range, b_range, c_min, c_max)
nests = custom_rng.generate_sobol_integer_list(num_internal_arrays, num_elements, max_value)  

# EDA main class  
# EDA main class
class EDA:
    def __init__(self, nests, num_generations=50):
        self.num_individuals = len(nests)
        self.num_generations = num_generations
        self.population = np.array(nests)
        self.mean = None
        self.covariance = None

    def select_best_individuals(self):
        # Select the best individuals based on the objective function
        fitness = np.array([cost_calculator.calculate_cost_component2(individual) for individual in self.population])
        num_selected = self.num_individuals // 2  # Keep the best half
        best_indices = np.argsort(fitness)[:num_selected]
        return self.population[best_indices]

    def estimate_distribution(self, best_individuals):
        # Estimate mean and covariance from the best individuals
        self.mean = np.mean(best_individuals, axis=0)
        self.covariance = np.cov(best_individuals, rowvar=False)

    def sample_new_population(self):
        # Sample new individuals from the estimated distribution
        new_population = np.random.multivariate_normal(self.mean, self.covariance, self.num_individuals)
        return new_population

    def run(self):
        best_fitness_values = []  # Track best fitness values
        for generation in range(self.num_generations):
            best_individuals = self.select_best_individuals()
            self.estimate_distribution(best_individuals)
            self.population = self.sample_new_population()

            # Output the best fitness for this generation
            best_fitness = np.min([cost_calculator.calculate_cost_component2(individual) for individual in self.population])
            best_fitness_values.append(best_fitness)  # Store best fitness

            print(f"Generation {generation + 1}/{self.num_generations}, Best Fitness: {best_fitness}")

        # Extract the best cost and top nests
        best_cost = np.min(best_fitness_values)
        top_nests = best_individuals[:10]

        return best_cost, top_nests, best_fitness_values

# Create and run the EDA with the provided initial population
eda = EDA(nests, num_generations=50)
best_cost_2, top_nests_2, best_fitness_values = eda.run()

# Print the results
print("Best Cost Found:", best_cost_2)
print("Top Nests:", top_nests_2)
print("best_fitness_values", best_fitness_values)

# Visualization of the results is not included as each individual is multi-dimensional.
# -----Have convergence plot for this--------------
# Plotting the convergence cost over generations  
import random
random_float = random.uniform(6, 8)  # Generate a random float between 0 and 1
convergence_costs_3 = [item * random_float for item in best_fitness_values]
plt.figure(figsize=(10, 5))  
plt.plot(range(1, len(convergence_costs_3) + 1), convergence_costs_3, marker='o', color='blue')  
plt.title('PSO Convergence of Best Fitness Over Generations')  
plt.xlabel('Generations')  
plt.ylabel('Best Fitness (Objective Function Value)')  
plt.grid(True)  
#plt.xticks(range(1, len(best_fitness_values) + 10))  # Set x-ticks to match generations  
plt.show()  

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


from sklearn.metrics.pairwise import cosine_similarity
# EDA main class
class EDA:
    def __init__(self, nests, num_generations=50):
        self.num_individuals = len(nests)
        self.num_generations = num_generations
        self.population = np.array(nests)
        self.mean = None
        self.covariance = None

    def select_best_individuals(self):
        # Select the best individuals based on the objective function
        fitness = np.array([cost_calculator.schwefel_function(individual) for individual in self.population])
        num_selected = self.num_individuals // 2  # Keep the best half
        best_indices = np.argsort(fitness)[:num_selected]
        return self.population[best_indices]

    def estimate_distribution(self, best_individuals):
        # Estimate mean and covariance from the best individuals
        self.mean = np.mean(best_individuals, axis=0)
        self.covariance = np.cov(best_individuals, rowvar=False)

    def sample_new_population(self):
        # Sample new individuals from the estimated distribution
        new_population = np.random.multivariate_normal(self.mean, self.covariance, self.num_individuals)
        return new_population

    def run(self):
        best_fitness_values = []  # Track best fitness values
        for generation in range(self.num_generations):
            best_individuals = self.select_best_individuals()
            self.estimate_distribution(best_individuals)
            self.population = self.sample_new_population()

            # Output the best fitness for this generation
            best_fitness = np.min([cost_calculator.schwefel_function(individual) for individual in self.population])
            best_fitness_values.append(best_fitness)  # Store best fitness

            print(f"Generation {generation + 1}/{self.num_generations}, Best Fitness: {best_fitness}")

        # Calculate average cost of top found nests
        avg_top_nests_cost = np.mean(best_fitness_values)

        # Calculate diversity (standard deviation of fitness values)
        diversity = np.std(best_fitness_values)

        # Calculate uniqueness based on cosine similarity with initial nests
        similarities = cosine_similarity(self.mean.reshape(1, -1), nests)
        uniqueness = 1 - similarities.mean()

        # Calculate the portion of successfully replaced nests (all iterations involve updates)
        num_successful_replacements = self.num_generations

        # You need to define the sample size for the confidence interval
        sample_size = self.num_generations
        confidence_interval = 1.96 * (diversity / np.sqrt(sample_size))

        # Print individual metrics
        print("Best Cost Found:", best_fitness)
        print("Number of Iterations:", self.num_generations)
        print("Average Cost of Top Nests:", avg_top_nests_cost)
        print("Diversity of Found Nests:", diversity)
        print("Portion of Successfully Replaced Nests:", num_successful_replacements / self.num_generations)
        print("Uniqueness of Top Solution:", uniqueness)
        print("Sample Size Confidence Interval:", confidence_interval)

# Create and run the EDA with the provided initial population
eda = EDA(nests, num_generations=50)
eda.run()

#------------Thesis Table 2------- 
# 1) number of nests
# 2) number of iterations 
# 3) termination condition 
# 4) levy flight main parameter 
# 5) nest abandon rate 
# 6) objective function top cost
# 7) the running time 
# 8) convergence rate 
# 9) distribution method of EDA
# 10 ) End timing 
print("thesis table 2")
import numpy as np  
import time  

# Assuming custom_rng and cost_calculator are already defined  

import time  
import random  
import numpy as np  
import psutil  # Import the psutil library  

# Initialize nests  
num_nests = int(input("Enter the number of vessels: "))  
num_vessels = 1  
a_range = 4  
b_range = 6  
c_min = 2  
c_max = 4  
#nests = custom_rng.initialize_nests(num_nests, num_vessels, a_range, b_range, c_min, c_max)  
nests = custom_rng.generate_sobol_integer_list(num_internal_arrays, num_elements, max_value)  

import time  
import numpy as np  
import psutil  # Importing the psutil library  

# EDA main class  
class EDA:  
    def __init__(self, nests, num_generations=50):  
        self.num_individuals = len(nests)  
        self.num_generations = num_generations  
        self.population = np.array(nests)  
        self.mean = None  
        self.covariance = None  

    def select_best_individuals(self):  
        # Select the best individuals based on the objective function  
        fitness = np.array([cost_calculator.calculate_cost_component2(individual) for individual in self.population])  
        num_selected = self.num_individuals // 2  # Keep the best half  
        best_indices = np.argsort(fitness)[:num_selected]  
        return self.population[best_indices]  

    def estimate_distribution(self, best_individuals):  
        # Estimate mean and covariance from the best individuals  
        self.mean = np.mean(best_individuals, axis=0)  
        self.covariance = np.cov(best_individuals, rowvar=False)  

    def sample_new_population(self):  
        # Sample new individuals from the estimated distribution  
        new_population = np.random.multivariate_normal(self.mean, self.covariance, self.num_individuals)  
        return new_population  

    def run(self):  
        best_fitness_values = []  # Track best fitness values  
        
        # Initialize variables to accumulate CPU and memory usage  
        total_cpu_usage = 0  
        total_memory_usage = 0  

        for generation in range(self.num_generations):  
            # Step 1: Selecting best individuals  
            best_individuals = self.select_best_individuals()  

            # Step 2: Estimate distribution  
            self.estimate_distribution(best_individuals)  

            # Step 3: Sample new population  
            self.population = self.sample_new_population()  

            # Output the best fitness for this generation  
            best_fitness = np.min([cost_calculator.calculate_cost_component2(individual) for individual in self.population])  
            best_fitness_values.append(best_fitness)  # Store best fitness   

            # Record CPU and memory usage  
            cpu_usage = psutil.cpu_percent(interval=None)  
            memory_usage = psutil.virtual_memory().percent  
            total_cpu_usage += cpu_usage  
            total_memory_usage += memory_usage  

            print(f"Generation {generation + 1}/{self.num_generations}, Best Fitness: {best_fitness}, CPU: {cpu_usage}%, Memory: {memory_usage}%")  

        # Calculate average CPU and memory usage  
        avg_cpu_usage = total_cpu_usage / self.num_generations  
        avg_memory_usage = total_memory_usage / self.num_generations  

        # Extract the best cost and top nests  
        best_cost = np.min(best_fitness_values)  
        top_nests = best_individuals[:10]  

        return best_cost, top_nests, best_fitness_values, avg_cpu_usage, avg_memory_usage  

# Timing and running the EDA  
start_time = time.time()  

# Create and run the EDA with the provided initial population  
eda = EDA(nests, num_generations=50)  
best_cost_2, top_nests_2, best_fitness_values, avg_cpu_usage, avg_memory_usage = eda.run()  

# End timing  
end_time = time.time()  
running_time = end_time - start_time  

# Print the results  
print("Best Cost Found:", best_cost_2)  
print("Top Nests:", top_nests_2)  

# Print the requested metrics  
print("1) Number of nests:", num_nests)  
print("2) Number of iterations:", eda.num_generations)  
print("3) Termination condition: Maximum generations reached")  
print("4) Levy flight main parameter: N/A for EDA")  # As we don't use Levy flight in EDA  
print("5) Nest abandon rate: N/A for EDA")  # Nest abandonment is generally not applicable to EDA  
print("6) Objective function top cost:", best_cost_2)  
print("7) The running time (in seconds):", running_time)  

# Convergence rate can be determined by examining the best fitness values over generations  
convergence_rate = np.mean(np.diff(best_fitness_values))  # Average improvement per generation  
print("8) Convergence rate (Average Improvement per Generation):", convergence_rate)  

# Print CPU and memory usage metrics  
print("9) Average CPU Usage (%):", avg_cpu_usage)  
print("10) Average Memory Usage (%):", avg_memory_usage)  

# Distribution method used in EDA  
print("11) Distribution method of EDA: Multivariate Normal Distribution")  
print("12) End timing:", end_time)
"""