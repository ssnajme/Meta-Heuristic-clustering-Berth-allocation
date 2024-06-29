import numpy as np
import random
import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


class CustomRandom:
    def __init__(self):
        pass

    def randint(self, a, b):
        """Return random integer in range [a, b], including both end points."""
        return self.randrange(a, b + 1)

    def randrange(self, start, stop=None, step=1):
        """Choose a random item from range(start, stop[, step])."""

        if stop is None:
            start, stop = 0, start

        width = stop - start
        if step == 1 and width > 0:
            return start + random.randint(0, width - 1)
        if step == 1:
            raise ValueError(f"empty range for randrange() ({start}, {stop}, {width})")

        n = (width + step - 1) // step

        if n <= 0:
            raise ValueError("empty range for randrange()")

        return start + step * random.randint(0, n - 1)

    def random_numbers(self, a, b, num):
        """Generate a specified number of random integers in range [a, b]."""
        return [self.randint(a, b) for _ in range(num)]

    def initialize_nests(self, num_nests, num_vessels, a_range, b_range, c_min , c_max):
        """Initialize nests with lists containing 9 items (C1, C2, C3, C4, Z, ETA, At, Dt, d)."""
        nests = []
        for _ in range(num_nests):
            nest = []
            for _ in range(num_vessels):
                # Generate random values for ETA, At, and Dt
                ETA = self.random_numbers(a_range, b_range, 1)[0]
                At = self.random_numbers(ETA, b_range, 1)[0]
                Dt = self.random_numbers(At, b_range, 1)[0]

                # Ensure ETA < At and At < Dt
                while At <= ETA or Dt <= At:
                    ETA = self.random_numbers(a_range, b_range, 1)[0]
                    At = self.random_numbers(ETA, b_range, 1)[0]
                    Dt = self.random_numbers(At, c_max, 1)[0]

                # Generate independent random numbers C1, C2, C3, C4
                #C1 = self.random_numbers(c_min, c_max, 1)[0]
                C2 = self.random_numbers(c_min, c_max, 1)[0]
                C3 = self.random_numbers(c_min, c_max, 1)[0]
                C4 = self.random_numbers(c_min, c_max, 1)[0]

                # Generate delay (d) such that At <<= d <<= Dt
                # maybe this value should change!
                d = self.random_numbers(Dt, c_max, 1)[0]

                nest.extend([ C2, C3, C4, ETA, At, Dt, d])
            nests.append(nest)
        return nests
    
    def latin_hypercube_sampling(bounds, num_samples, num_vars):
        samples = np.zeros((num_samples, num_vars), dtype=int)

        # Generate random samples within each interval of the Latin Hypercube
        for i in range(num_vars):
            step_size = max(1, (bounds[i][1] - bounds[i][0]) // num_samples)
            for j in range(num_samples):
                samples[j, i] = random.randint(bounds[i][0] + j * step_size, min(bounds[i][0] + (j + 1) * step_size - 1, bounds[i][1]))

        # Shuffle the samples within each variable to ensure randomness
        for i in range(num_vars):
            np.random.shuffle(samples[:, i])

        return samples
    
    def sobol_seq(dim, n):
        def sobol_generate(n, dim):
            s = [[0] for _ in range(n)]
            for i in range(n):
                s[i] = [0] * dim
                for j in range(dim):
                    s[i][j] = 0
                    k = i
                    while k > 0:
                        k, t = divmod(k, 2)
                        s[i][j] += (1 << t) * (1 + (j * 2 ** t) % 3)
            return s

        s = sobol_generate(n, dim)
        return s



# Example usage:
custom_rng = CustomRandom()
#num_nests = 
num_nests = int(input("Enter the number of vessels: "))
num_vessels = 1
a_range = 50
b_range = 100
c_min = 20
c_max = 350
#c2_max = 300
result = custom_rng.initialize_nests(num_nests, num_vessels, a_range, b_range, c_min, c_max)
print(result)


""" 
################ for latin_hypercube_sampling method#### 
# Define the search space bounds for each decision variable
bounds = [(0, 10), (-5, 5), (1, 100), (10, 20), (0, 50), (5, 15), (1, 10)]

# Specify the number of samples (population size) to generate using LHS
num_samples = 10
num_vars = 7

# Perform Latin Hypercube Sampling to generate initial population
lhs_samples = custom_rng.latin_hypercube_sampling(bounds, num_samples, num_vars)

# Print the generated initial population of solutions
print("Initial Population Generated using Latin Hypercube Sampling:")
for i in range(num_samples):
    print(f"Solution {i+1}: {lhs_samples[i]}")


################ for sobol method######### 
# Number of dimensions
num_dimensions = 7

# Number of points to generate
num_points = 10

# Generate Sobol sequence
sobol_sequence = custom_rng.sobol_seq(num_dimensions, num_points)

# Shuffle the Sobol sequence for randomization
random.shuffle(sobol_sequence)

# Print the shuffled Sobol sequence
print(sobol_sequence)
"""
