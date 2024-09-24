# --- First plot ----
#####################
import numpy as np  
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  

# ----- 3D plot -- decision space ----------------
# -------------------------------------------------
# Parameters (example values, you can adjust)  
l = 9  # Number of terms in the summation (9 variables)  
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
        cost = (  
            c1k[k] * Z_k[k] +  
            c2k[k] * np.maximum(0, ETA_k - At_k) +  # Penalty for late arrival  
            c3k[k] * np.maximum(0, At_k - ETA_k) +  # Penalty for early arrival  
            c4k[k] * np.maximum(0, Dt_k - d_k[k])   # Penalty for not meeting the fixed target  
        )  
        total_cost += cost  # Aggregate the cost for total cost  
    return total_cost  # Return the final calculated cost  

# Create ranges for ETA_k and At_k values, and define fixed Dt_k targets  
ETA_k_vals = np.linspace(0, 10, 400)  # Generate 400 values between 0 and 10 for ETA_k  
At_k_vals = np.linspace(0, 10, 400)    # Generate 400 values between 0 and 10 for At_k  
fixed_Dt_k_vals = np.linspace(0, 10, 4)  # Define 4 fixed target values for Dt_k  

# Create a meshgrid for ETA_k and At_k for plotting  
ETA_k_grid, At_k_grid = np.meshgrid(ETA_k_vals, At_k_vals)  

# Plotting  
fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Create a 2x2 grid of subplots  

# Loop through the fixed target values to create individual plots  
for index, Dt_k in enumerate(fixed_Dt_k_vals):  
    # Calculate the objective function values for the current fixed Dt_k  
    min_cost_values = MinCostVessels(ETA_k_grid, At_k_grid, Dt_k)  

    # Create a contour plot for the current Dt_k value  
    ax = axs[index // 2, index % 2]  
    contour = ax.contourf(ETA_k_grid, At_k_grid, min_cost_values, levels=50, cmap='plasma', alpha=0.7)  
    
    # Add scatter samples to the plot  
    sample_indices = np.random.choice(len(ETA_k_vals) * len(At_k_vals), size=100, replace=False)  
    sample_ETAs = ETA_k_grid.flatten()[sample_indices]  
    sample_Ats = At_k_grid.flatten()[sample_indices]  
    sample_costs = min_cost_values.flatten()[sample_indices]  
    
    ax.scatter(sample_ETAs, sample_Ats, c='black', marker='o', label='Sampled Data', s=30)  

    # Setting titles and labels for clarity of each subplot  
    ax.set_title(f'Dt_k = {Dt_k:.2f}')  # Title indicates the current fixed target value  
    ax.set_xlabel('ETA_k')  # Label for ETA_k axis  
    ax.set_ylabel('At_k')   # Label for At_k axis  
    ax.legend()  # Add legend to the plot  

    # Add color bar for reference on cost values  
    plt.colorbar(contour, ax=ax, shrink=0.5, aspect=10)  

plt.tight_layout()  # Adjust layout to prevent overlap  
plt.show()  # Display the plots
 
# ------------------------------------------------------------
import numpy as np  
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm  
from matplotlib.colors import Normalize  

# Create a grid of points  
x = np.linspace(0, 2, 10)  
y = np.linspace(0, 2, 10)  
z = np.linspace(0, 2, 10)  
X, Y, Z = np.meshgrid(x, y, z)  

# Define the vector field (example: a simple radial field)  
U = X  
V = Y  
W = Z  

# Calculate the magnitude of the vectors for coloring  
magnitude = np.sqrt(U**2 + V**2 + W**2)  

# Normalize the magnitude for color mapping  
norm = Normalize(vmin=magnitude.min(), vmax=magnitude.max())  
colors = cm.viridis(norm(magnitude.flatten()))  

# Create a figure  
fig = plt.figure()  
ax = fig.add_subplot(111, projection='3d')  

# Create a quiver plot with larger arrows  
quiver = ax.quiver(X.flatten(), Y.flatten(), Z.flatten(),   
                    U.flatten(), V.flatten(), W.flatten(),   
                    length=0.2, color=colors)  

# Set labels  
ax.set_xlabel('X axis')  
ax.set_ylabel('Y axis')  
ax.set_zlabel('Z axis')  

# Add a color bar  
mappable = cm.ScalarMappable(cmap=cm.viridis, norm=norm)  
mappable.set_array(magnitude.flatten())  # Flatten for the color bar  
plt.colorbar(mappable, ax=ax, label='Magnitude')  

# Show the plot  
plt.show()
#----------------------------------------------------------------

