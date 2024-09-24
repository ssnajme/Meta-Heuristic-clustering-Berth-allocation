import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

#----- the default objective function is the pareto itself
def calculate_cost_component1(self, params):
        C1, Z = params
        return self.weights[0] * C1 * Z

def calculate_cost_component2(self, params):
        C2, C3, C4, ETA, At, Dt, d = params

        cost_components = np.array([
            self.weights[1] * C2 * np.maximum(ETA - At, 0),
            self.weights[2] * C3 * np.maximum(At - ETA, 0),
            self.weights[3] * C4 * np.maximum(Dt - d, 0),
        ])

        total_cost = np.sum(cost_components)

        if self.T_start <= At <= self.T_end or self.T_start <= Dt <= self.T_end:
            total_cost += self.penalty_weight * (max(0, self.T_start - At) + max(0, Dt - self.T_end))

        return total_cost

def pareto_function(self, params):
        C1, C2, C3, C4, Z, ETA, At, Dt, d = params

        cost_component1 = self.calculate_cost_component1([C1, Z])
        cost_component2 = self.calculate_cost_component2([C2, C3, C4, ETA, At, Dt, d])

        #total_cost = cost_component1 + cost_component2 + self.num_vessels * self.some_coefficient
        total_cost = cost_component1 + cost_component2 

        return total_cost

### ---- this is a sample code for passing the costs ------ 
### --- this produces 3 charts -------better pareto referencing ----
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

# Generate random data for demonstration
np.random.seed(0)
data = np.random.rand(100, 3)

# Calculate Pareto front
def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] <= c, axis=1)
            is_efficient[i] = True
    return is_efficient

pareto_mask = is_pareto_efficient(data)
pareto_front = data[pareto_mask]

# Generate reference points
reference_points = np.random.rand(10, 3)

# Calculate convex hull for Pareto front
hull = ConvexHull(pareto_front)

# Plotting
fig = plt.figure(figsize=(18, 6))

# First angle
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c='gray', label='Data')
ax1.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], c='blue', label='Pareto front')
ax1.scatter(reference_points[:, 0], reference_points[:, 1], reference_points[:, 2], c='orange', label='Reference points')
ax1.plot_trisurf(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], triangles=hull.simplices, alpha=0.2, color='cyan')
ax1.set_xlabel('f1')
ax1.set_ylabel('f2')
ax1.set_zlabel('f3')
ax1.view_init(elev=20, azim=30)
ax1.legend()

# Second angle
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(data[:, 0], data[:, 1], data[:, 2], c='gray', label='Data')
ax2.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], c='blue', label='Pareto front')
ax2.scatter(reference_points[:, 0], reference_points[:, 1], reference_points[:, 2], c='orange', label='Reference points')
ax2.plot_trisurf(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], triangles=hull.simplices, alpha=0.2, color='cyan')
ax2.set_xlabel('f1')
ax2.set_ylabel('f2')
ax2.set_zlabel('f3')
ax2.view_init(elev=20, azim=120)
ax2.legend()

# Third angle
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(data[:, 0], data[:, 1], data[:, 2], c='gray', label='Data')
ax3.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], c='blue', label='Pareto front')
ax3.scatter(reference_points[:, 0], reference_points[:, 1], reference_points[:, 2], c='orange', label='Reference points')
ax3.plot_trisurf(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], triangles=hull.simplices, alpha=0.2, color='cyan')
ax3.set_xlabel('f1')
ax3.set_ylabel('f2')
ax3.set_zlabel('f3')
ax3.view_init(elev=20, azim=210)
ax3.legend()

plt.show()

# ---- this is the base sample ------
# -----------------------------------
np.random.seed(0)
data = np.random.rand(100, 3)

# Calculate Pareto front
def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] <= c, axis=1)
            is_efficient[i] = True
    return is_efficient

pareto_mask = is_pareto_efficient(data)
pareto_front = data[pareto_mask]

# Generate reference points
reference_points = np.random.rand(10, 3)

# Calculate convex hull for Pareto front
hull = ConvexHull(pareto_front)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='gray', label='Data')
ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], c='blue', label='Pareto front')
ax.scatter(reference_points[:, 0], reference_points[:, 1], reference_points[:, 2], c='orange', label='Reference points')

# Plot convex hull
for simplex in hull.simplices:
    ax.plot(pareto_front[simplex, 0], pareto_front[simplex, 1], pareto_front[simplex, 2], 'k-', alpha=0.5)

# Plot convex surface
ax.plot_trisurf(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], triangles=hull.simplices, alpha=0.2, color='cyan')

ax.set_xlabel('f1')
ax.set_ylabel('f2')
ax.set_zlabel('f3')
ax.legend()
plt.show()

# --- just work on passing the costs here maybe ---  
# --- Modify the first plot ---------------------------