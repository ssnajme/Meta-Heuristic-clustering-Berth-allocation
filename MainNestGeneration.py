import numpy as np
import random
import warnings
from scipy.stats import qmc 

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


    def initialize_nests(self, num_nests, num_vessels, a_range, b_range, c_min, c_max):
        """Initialize nests with lists containing 9 items (C1, C2, C3, C4, Z, ETA, At, Dt, d)."""
        nests = []
        for _ in range(num_nests):
            nest = []
            for _ in range(num_vessels):
                # Generate random time slots (in 4-hour intervals within a span of 2 days)
                slot_start = random.randint(0, 11)  # 12 slots in 2 days (0-indexed)
                ETA = a_range + (slot_start * 4)
                At = ETA + random.randint(1, 4)  # Assuming vessels arrive within 1-4 hours after ETA
                #Dt = At + random.randint(1, 4)  # Assuming vessels depart within 1-4 hours after arrival
                
                #  Ensure that the difference between Dt and At is large
                min_dt_offset = 20  # Minimum difference between Dt and At
                max_dt_offset = 34  # Maximum difference between Dt and At

                # Generate Dt such that it is at least 'min_dt_offset' hours after At
                # The range from At to Dt will now be [At+min_dt_offset, At+max_dt_offset]
                Dt = At + random.randint(min_dt_offset, max_dt_offset)

                # Generate independent random numbers C2, C3, C4
                C2 = random.randint(c_min, c_max)
                C3 = random.randint(c_min, c_max)
                C4 = random.randint(c_min, c_max)

                # Generate delay (d) such that At <= d <= Dt
                d = Dt + random.randint(0,4)

                nest.extend([C2, C3, C4, ETA, At, Dt, d])
            nests.append(nest)
        return nests
    
    def latin_hypercube_sampling(self, bounds, num_samples, num_vars):
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
    
    def generate_sobol_integer_list(self, num_arrays, num_elements, max_value):  
            # Initialize the Sobol engine  
            sobol_engine = qmc.Sobol(d=num_elements, scramble=True)  

            # Generate Sobol sequence points between 0 to 1  
            sobol_samples = sobol_engine.random(n=num_arrays)  

            # Scale the samples to integers within the specified range  
            sobol_integer_array = (sobol_samples * max_value).astype(int).tolist()  

            # Ensure increasing properties - adapt sample values for increasing patterns  
            for i in range(num_arrays):  
                for j in range(num_elements):  
                    if j > 0 and sobol_integer_array[i][j] <= sobol_integer_array[i][j - 1]:  
                        sobol_integer_array[i][j] = sobol_integer_array[i][j - 1] + 1  

            return sobol_integer_array  


custom_rng = CustomRandom()
num_nests = int(input("Enter the number of vessels: "))
num_vessels = 1
a_range = 4
b_range = 6
c_min = 2
c_max = 4
#Nests = custom_rng.initialize_nests(num_nests, num_vessels, a_range, b_range, c_min, c_max)

#print(Nests)


#---- second data generation method -- for latin_hypercube_sampling method---# 
# Parameters  
num_dimensions = 7  
num_points = 20 
bounds = [(0, 66), (-5, 5), (1, 100), (10, 20), (0, 50), (5, 15), (1, 66)]  
num_samples = num_points  
num_vars = num_dimensions  



# Generate Sobol sequence samples  
# Specify the number of internal arrays and the number of elements in each  
num_internal_arrays = 66  # For example, 10 internal arrays  
num_elements = 7          # Each internal array will have this many elements  
max_value = 20            # Maximum value for the integers  

# Generate the Sobol sequence based 2D integer array  
resulting_array = custom_rng.generate_sobol_integer_list(num_internal_arrays, num_elements, max_value)  

print("Generated Sobol Integer 2D Array:")  
print(resulting_array)  
