import matplotlib.pyplot as plt  
import numpy as np  
"""
# Data  
efficiency = [74.65, 67.61, 83.23, 84.56]  
cost = [0.95, 5.30, 4.50, 6.45]  
executions = [55, 60, 70, 75]  
vessels = [6, 9, 11, 15]  

# X-axis positions  
x = np.arange(len(executions))  

# Bar width  
width = 0.35  

# Create subplots  
fig, ax1 = plt.subplots(1, 2, figsize=(12, 6))  

# Bar chart for Efficiency vs Cost  
ax1[0].bar(x - width/2, efficiency, width, label='Efficiency (%)', color='skyblue')  
ax1[0].bar(x + width/2, cost, width, label='Average Cost ($)', color='salmon')  
ax1[0].set_title('Efficiency vs Average Cost')  
ax1[0].set_xlabel('Number of Executions')  
ax1[0].set_ylabel('Values')  
ax1[0].set_xticks(x)  
ax1[0].set_xticklabels(executions)  
ax1[0].legend()  
ax1[0].grid(axis='y', linestyle='--', alpha=0.7)  

# Bar chart for Efficiency vs Number of Vessels  
ax1[1].bar(x - width/2, efficiency, width, label='Efficiency (%)', color='skyblue')  
ax1[1].bar(x + width/2, vessels, width, label='Number of Vessels', color='lightgreen')  
ax1[1].set_title('Efficiency vs Number of Vessels')  
ax1[1].set_xlabel('Number of Executions')  
ax1[1].set_ylabel('Values')  
ax1[1].set_xticks(x)  
ax1[1].set_xticklabels(executions)  
ax1[1].legend()  
ax1[1].grid(axis='y', linestyle='--', alpha=0.7)  

# Show the plot  
plt.tight_layout()  
plt.show()
""" 

import matplotlib.pyplot as plt  
import numpy as np  

""" 

# Data  
efficiency = [76.68, 61.94, 67.52, 85.69]  
executions = [55, 65, 75, 80]  
time = [3.87564, 4.3416, 4.8356, 6.2637]  
cost = [2.4738, 3.1347, 3.0828, 3.5018]  
berths = [3, 5, 7, 10]  

# X-axis positions  
x = np.arange(len(executions))  

# Bar width  
width = 0.2  

# Creating the figure and axes  
fig, ax = plt.subplots(figsize=(10, 6))  

# Bar charts for Efficiency  
bars1 = ax.bar(x - 1.5 * width, efficiency, width, label='Efficiency (%)', color='skyblue')  
# Bar charts for Time  
bars2 = ax.bar(x - 0.5 * width, time, width, label='Time (hours)', color='salmon')  
# Bar charts for Cost  
bars3 = ax.bar(x + 0.5 * width, cost, width, label='Cost ($)', color='lightgreen')  
# Bar charts for Number of Berths  
bars4 = ax.bar(x + 1.5 * width, berths, width, label='Number of Berths', color='orange')  

# Adding titles and labels  
ax.set_title('Comparison of Vessel Metrics')  
ax.set_xlabel('Number of Executions')  
ax.set_ylabel('Values')  
ax.set_xticks(x)  
ax.set_xticklabels(executions)  

# Adding a legend  
ax.legend()  

# Display grid lines  
ax.yaxis.grid(True, linestyle='--', alpha=0.7)  

# Adding labels on top of bars  
def add_labels(bars):  
    for bar in bars:  
        yval = bar.get_height()  
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', va='bottom', ha='center')  

add_labels(bars1)  
add_labels(bars2)  
add_labels(bars3)  
add_labels(bars4)  

# Show the plot  
plt.tight_layout()  
plt.show()
""" 


""" 
# Data extracted from the table  
efficiency = [83.33, 51.39, 53.22, 69.41]  
number_of_vessels = [6, 9, 11, 15]  
time_window = [1.34525, 1.78335, 2.34386, 2.1563]  
cost = [3.340461, 2.548743, 6.357668, 2.382546]  
number_of_berths = [12, 21, 17, 10]  

# X-axis positions  
x = np.arange(len(efficiency))  

# Bar width  
width = 0.15  

# Creating the figure and axes  
fig, ax = plt.subplots(figsize=(10, 6))  

# Bar charts for each metric  
bars1 = ax.bar(x - 2 * width, number_of_berths, width, label='Number of Berths', color='skyblue')  
bars2 = ax.bar(x - width, number_of_vessels, width, label='Number of Vessels', color='salmon')  
bars3 = ax.bar(x, time_window, width, label='Time Window (hours)', color='lightgreen')  
bars4 = ax.bar(x + width, efficiency, width, label='Efficiency (%)', color='orange')  
bars5 = ax.bar(x + 2 * width, cost, width, label='Cost ($)', color='lightcoral')  

# Adding titles and labels  
ax.set_title('Comparison of Vessel Metrics')  
ax.set_xlabel('Execution Index')  
ax.set_ylabel('Values')  
ax.set_xticks(x)  
ax.set_xticklabels([1, 2, 3, 4])  # Using indices for clarity  

# Adding a legend  
ax.legend()  

# Display grid lines  
ax.yaxis.grid(True, linestyle='--', alpha=0.7)  

# Adding labels on top of bars  
def add_labels(bars):  
    for bar in bars:  
        yval = bar.get_height()  
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', va='bottom', ha='center')  

add_labels(bars1)  
add_labels(bars2)  
add_labels(bars3)  
add_labels(bars4)  
add_labels(bars5)  

# Show the plot  
plt.tight_layout()  
plt.show()
""" 

""" 
# Data extracted from the table  
weather_conditions = ['Condition 1', 'Condition 2', 'Condition 3', 'Condition 4', 'Condition 5', 'Condition 6']  
efficiency = [76.38, 34.27, 38.63, 48.79, 36.48, 81.76]  
risk_factor = [3.985, 10.759, 6.236, 5.376, 7.291, 2.674]  
number_of_vessels = [7, 7, 7, 11, 10, 10]  
time_window = [12, 15, 17, 10, 24, 8]  # Corrected to 5 entries  
number_of_berths = [3, 3, 3, 4, 4, 4]  

# X-axis positions  
x = np.arange(len(weather_conditions))  

# Bar width  
width = 0.15  

# Creating the figure and axes  
fig, ax = plt.subplots(figsize=(12, 6))  

# Bar charts for each metric  
bars1 = ax.bar(x - 2 * width, number_of_berths, width, label='Number of Berths', color='skyblue')  
bars2 = ax.bar(x - width, number_of_vessels, width, label='Number of Vessels', color='salmon')  
bars3 = ax.bar(x, time_window, width, label='Time Window (hours)', color='lightgreen')  
bars4 = ax.bar(x + width, efficiency, width, label='Efficiency (%)', color='orange')  
bars5 = ax.bar(x + 2 * width, risk_factor, width, label='Risk Factor', color='lightcoral')  

# Adding titles and labels  
ax.set_title('Comparison of berths Metrics by Weather Condition')  
ax.set_xlabel('Weather Conditions')  
ax.set_ylabel('Values')  
ax.set_xticks(x)  
ax.set_xticklabels(weather_conditions)  

# Adding a legend  
ax.legend()  

# Display grid lines  
ax.yaxis.grid(True, linestyle='--', alpha=0.7)  

# Adding labels on top of bars  
def add_labels(bars):  
    for bar in bars:  
        yval = bar.get_height()  
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', va='bottom', ha='center')  

add_labels(bars1)  
add_labels(bars2)  
add_labels(bars3)  
add_labels(bars4)  
add_labels(bars5)  

# Show the plot  
plt.tight_layout()  
plt.show()


import matplotlib.pyplot as plt  
import numpy as np  

# Data extracted from the table  
berth_efficiency = [66.13, 76.28, 70.35, 48.32]  
number_of_vessels = [7, 7, 10, 10]  
number_of_berths = [3,3, 4, 4]  
risk_factor = [6.411, 2.731, 3.784, 8.634]  
tide_levels = ['0 < 0.63 < 1', '0 < 0.16 < 1', '0 < 0.23 < 1', '0 < 0.84 < 1']  

# X-axis positions  
x = np.arange(len(tide_levels))  

# Bar width  
width = 0.2  

# Creating the figure and axes  
fig, ax = plt.subplots(figsize=(12, 6))  

# Bar charts for each metric  
bars1 = ax.bar(x - 1.5 * width, berth_efficiency, width, label='Berth Efficiency (%)', color='skyblue')  
bars2 = ax.bar(x - 0.5 * width, number_of_vessels, width, label='Number of Vessels', color='salmon')  
bars3 = ax.bar(x + 0.5 * width, number_of_berths, width, label='Number of Berths', color='lightgreen')  
bars4 = ax.bar(x + 1.5 * width, risk_factor, width, label='Risk Factor', color='lightcoral')  

# Adding titles and labels  
ax.set_title('Berth Efficiency, Vessels, Berths and Risk Factor Based on Tide Level')  
ax.set_xlabel('Tide Level')  
ax.set_ylabel('Values')  
ax.set_xticks(x)  
ax.set_xticklabels(tide_levels)  

# Adding a legend  
ax.legend()  

# Adding a grid  
ax.yaxis.grid(True, linestyle='--', alpha=0.7)  

# Function to add labels on top of bars  
def add_labels(bars):  
    for bar in bars:  
        yval = bar.get_height()  
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', va='bottom', ha='center')  

# Adding labels to each set of bars  
add_labels(bars1)  
add_labels(bars2)  
add_labels(bars3)  
add_labels(bars4)  

# Show the plot  
plt.tight_layout()  
plt.show()
""" 

import matplotlib.pyplot as plt  
import numpy as np  

# Sample data for four algorithms (replace these with your actual convergence values)  
iterations = np.arange(1, 11)  # Assuming 10 iterations for each algorithm  
algorithm1 = [100, 90, 80, 70, 65, 60, 58, 55, 53, 50]  # Example values  
algorithm2 = [100, 95, 85, 76, 70, 65, 60, 58, 54, 50]  # Example values  
algorithm3 = [100, 92, 88, 85, 82, 80, 78, 75, 73, 70]  # Example values  
algorithm4 = [100, 97, 90, 85, 80, 78, 75, 74, 72, 71]  # Example values  

# Creating the plot  
plt.figure(figsize=(12, 6))  

# Plotting each algorithm's convergence (decreasing)  
plt.plot(iterations, algorithm1, marker='o', label='Algorithm 1', color='blue')  
plt.plot(iterations, algorithm2, marker='o', label='Algorithm 2', color='orange')  
plt.plot(iterations, algorithm3, marker='o', label='Algorithm 3', color='green')  
plt.plot(iterations, algorithm4, marker='o', label='Algorithm 4', color='red')  

# Adding titles and labels  
plt.title('Convergence Plots for Four Algorithms')  
plt.xlabel('Iterations')  
plt.ylabel('Objective Function Value (Lower is Better)')  
plt.xticks(iterations)  
plt.grid(True)  

# Adding a legend  
plt.legend()  

# Show the plot  
plt.tight_layout()  
plt.show()