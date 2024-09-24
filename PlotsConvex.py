import numpy as np  
import matplotlib.pyplot as plt  

# ----- why this plot is not curved? ----------
# ---------------------------------------------

# Parameters (example values, you can adjust)  
l = 5  # number of terms in the summation  
c1k = np.random.rand(l)  # Random coefficients for illustration  
c2k = np.random.rand(l)  
c3k = np.random.rand(l)  
c4k = np.random.rand(l)  

Z_k = np.random.rand(l)  # Example values for Z_k, which impact cost calculation  
d_k = np.random.rand(l)  # Random fixed target values for each Dt_k scenario  

# Objective function to calculate minimum cost based on ETA_k, At_k, and Dt_k  
def MinCostVessels(ETA_k, At_k, Dt_k):  
    total_cost = 0  # Initialize the total cost to zero  
    for k in range(l):  # Iterate over the number of terms  
        # Cost calculation based on deviations from ETA, At, and fixed target Dt_k  
        cost = (c1k[k] * Z_k[k] +   
                 c2k[k] * np.maximum(0, ETA_k - At_k) +  # Penalty for late arrival  
                 c3k[k] * np.maximum(0, At_k - ETA_k) +  # Penalty for early arrival  
                 c4k[k] * np.maximum(0, Dt_k - d_k[k]))  # Penalty for not meeting the fixed target  
        total_cost += cost  # Aggregate the cost for total cost  
    return total_cost  # Return the final calculated cost  

# Create ranges for ETA_k and At_k values, and define fixed Dt_k targets  
ETA_k_vals = np.linspace(0, 10, 400)  # Generate 400 values between 0 and 10 for ETA_k  
At_k_vals = np.linspace(0, 10, 400)  # Generate 400 values between 0 and 10 for At_k  
fixed_Dt_k_vals = np.linspace(0, 10, 4)  # Define 4 fixed target values for Dt_k  

# Create a meshgrid for ETA_k and At_k for plotting  
ETA_k_grid, At_k_grid = np.meshgrid(ETA_k_vals, At_k_vals)  

# Plotting  
fig = plt.figure(figsize=(12, 10))  # Define the figure size  

# Loop through the fixed target values to create individual plots  
for index, Dt_k in enumerate(fixed_Dt_k_vals):  
    # Calculate the objective function values for the current fixed Dt_k  
    min_cost_values = MinCostVessels(ETA_k_grid, At_k_grid, Dt_k)  

    # Create a 3D subplot for the current Dt_k value  
    ax = fig.add_subplot(2, 2, index + 1, projection='3d')  # Create 2x2 grid of subplots  
    surf = ax.plot_surface(ETA_k_grid, At_k_grid, min_cost_values, cmap='plasma', edgecolor='none', antialiased=True)  

    # Setting titles and labels for clarity of each subplot  
    ax.set_title(f'Dt_k = {Dt_k:.2f}')  # Title indicates the current fixed target value  
    ax.set_xlabel('ETA_k')  # Label for ETA_k axis  
    ax.set_ylabel('At_k')  # Label for At_k axis  
    ax.set_zlabel('MinCostVessels')  # Label for the cost function  
    ax.view_init(elev=30, azim=240)  # Adjust the elevation and azimuth for better visualization  

    # Improve layout of grid lines for a rounder look  
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))  # Limit ticks on the x-axis  
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))  # Limit ticks on the y-axis  
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))  # Limit ticks on the z-axis  

# Adding color bar to the last subplot for reference on cost values  
plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)  # Display a color bar to indicate cost scale  
plt.tight_layout()  # Adjust layout to prevent overlap  
plt.show()  # Display the plots
#------------------------------------------------------------
