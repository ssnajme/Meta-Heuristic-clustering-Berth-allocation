import numpy as np  
import matplotlib.pyplot as plt  

# ---- For Now -------------------------------------------------------
# ------ change the format and color, those are off here ----
# ------ for each algorithm separately 



# --- writing a pareto function based on the current information is
# --- a story of it's own
# --- we want to plot the decision boundaries ------------------------
# --- have a plot for mapping decision space to the objective space --
# --- plot this based on the nests and their costs only ----- 

# --- Objective space - Plot 2 ----------------------------------------
import numpy as np  
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  

# Define the cost function  
def calculate_cost_component2(params):  
    C2, C3, C4, ETA, At, Dt, d = params  

    weights = [0, 0.2, 0.3, 0.5]  # Define weights for cost components  
    penalty_weight = 0.1  # Define penalty weight  

    cost_components = np.array([  
        weights[1] * C2 * np.maximum(ETA - At, 0),  
        weights[2] * C3 * np.maximum(At - ETA, 0),  
        weights[3] * C4 * np.maximum(Dt - d, 0),  
    ])  

    total_cost = np.sum(cost_components)  

    if 0 <= At <= 10 or 0 <= Dt <= 10:  # Define condition for penalty calculation  
        total_cost += penalty_weight * (max(0, 0 - At) + max(0, Dt - 10))  

    return total_cost  

# Create a grid of points for At and Dt  
At = np.linspace(0, 10, 100)  # Range for At  
Dt = np.linspace(0, 10, 100)  # Range for Dt  
At_grid, Dt_grid = np.meshgrid(At, Dt)  

# Fix other parameters  
C2, C3, C4 = 1, 1, 1  # Fixed values for cost components  
ETA = 5  
d = 5  # Fixed value for d  

# Compute costs for the grid  
Z = np.zeros_like(At_grid)  
for i in range(At_grid.shape[0]):  
    for j in range(At_grid.shape[1]):  
        params = (C2, C3, C4, ETA, At_grid[i, j], Dt_grid[i, j], d)  
        Z[i, j] = calculate_cost_component2(params)  

# Create a figure  
fig = plt.figure(figsize=(15, 5))  

# Center Planes  
ax1 = fig.add_subplot(131, projection='3d')  
ax1.plot_surface(At_grid, Dt_grid, Z, cmap='viridis', alpha=0.7)  
ax1.set_title('Center Planes')  
ax1.set_xlabel('At')  
ax1.set_ylabel('Dt')  
ax1.set_zlabel('Total Cost')  

# Back Planes  
ax2 = fig.add_subplot(132, projection='3d')  
ax2.plot_surface(At_grid, Dt_grid, Z, cmap='plasma', alpha=0.7)  
ax2.view_init(elev=30, azim=150)  
ax2.set_title('Back Planes')  
ax2.set_xlabel('At')  
ax2.set_ylabel('Dt')  
ax2.set_zlabel('Total Cost')  

# Diagonal Stacked Planes  
ax3 = fig.add_subplot(133, projection='3d')  
ax3.plot_surface(At_grid, Dt_grid, Z, cmap='inferno', alpha=0.7)  
ax3.view_init(elev=45, azim=45)  
ax3.set_title('Diagonal Stacked Planes')  
ax3.set_xlabel('At')  
ax3.set_ylabel('Dt')  
ax3.set_zlabel('Total Cost')  

# Show the plot  
plt.tight_layout()  
plt.show()

