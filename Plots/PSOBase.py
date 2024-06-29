import numpy as np
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, num_dimensions, c_min, c_max):
        self.position = np.random.uniform(c_min, c_max, num_dimensions)
        self.velocity = np.random.uniform(-1, 1, num_dimensions)
        self.best_position = self.position
        self.best_cost = float('inf')

class ParticleSwarmOptimization:
    def __init__(self, num_particles, num_dimensions, num_iterations, a_range, b_range, c_min, c_max, Lambda, step_size):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.num_iterations = num_iterations
        self.a_range = a_range
        self.b_range = b_range
        self.c_min = c_min
        self.c_max = c_max
        self.Lambda = Lambda
        self.step_size = step_size
        self.particles = [Particle(num_dimensions, c_min, c_max) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_cost = float('inf')

    def optimize(self):
        particle_positions_history = []
        
        for _ in range(self.num_iterations):
            current_positions = [particle.position for particle in self.particles]
            particle_positions_history.append(current_positions)
            
            for particle in self.particles:
                cost = cost_calculator.calculate_cost_component2(particle.position)
                if cost < particle.best_cost:
                    particle.best_cost = cost
                    particle.best_position = particle.position
                
                if cost < self.global_best_cost:
                    self.global_best_cost = cost
                    self.global_best_position = particle.position
                
                cognitive_weight = 1.5
                social_weight = 1.5
                inertia_weight = 0.5
                r1 = np.random.rand(self.num_dimensions)
                r2 = np.random.rand(self.num_dimensions)
                particle.velocity = (inertia_weight * particle.velocity +
                                     cognitive_weight * r1 * (particle.best_position - particle.position) +
                                     social_weight * r2 * (self.global_best_position - particle.position))
                
                particle.position += self.step_size * particle.velocity

        return particle_positions_history, self.global_best_position, self.global_best_cost

# Example usage:
num_particles = 30
num_dimensions = 3
num_iterations = 30
a_range = 0
b_range = 0
c_min = 0
c_max = 100
Lambda = 1.5
step_size = 0.1

pso = ParticleSwarmOptimization(num_particles, num_dimensions, num_iterations, a_range, b_range, c_min, c_max, Lambda, step_size)
particle_positions_history, best_position, best_cost = pso.optimize()

print("Global Best Position:")
print(best_position)
print("Global Best Cost:")
print(best_cost)

# Visualization
for i in range(len(particle_positions_history)):
    positions = np.array(particle_positions_history[i])
    plt.scatter(positions[:, 0], positions[:, 1], c='b', alpha=i / len(particle_positions_history))

plt.scatter(best_position[0], best_position[1], c='r', label='Global Best Position')
plt.legend()
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Particle Swarm Optimization Search Space Visualization')
plt.show()