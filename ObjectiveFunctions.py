import numpy as np
import random 

class CostCalculator:
    def __init__(self, weights, penalty_weight, num_vessels, some_coefficient, objective_function_type='default', T_start=0, T_end=10):
        self.weights = weights
        self.penalty_weight = penalty_weight
        self.num_vessels = num_vessels
        self.some_coefficient = some_coefficient
        self.objective_function_type = objective_function_type
        self.T_start = T_start
        self.T_end = T_end
        
        # Map objective function types to methods  
        self.objective_functions = {  
            'default': self.default_objective_function,  
            'calculate_cost_component1': self.calculate_cost_component1,  
            'calculate_cost_component2': self.calculate_cost_component2,  
            'rosenbrock': self.rosenbrock_function,  
            'ackley': self.ackley_function,  
            'rastrigin': self.rastrigin_function,  
            'griewank': self.griewank_function,  
            'schwefel': self.schwefel_function,  
        }  
        

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

        total_cost = np.sum(cost_components) # --- this is for the sigma value 

        if self.T_start <= At <= self.T_end or self.T_start <= Dt <= self.T_end:
            total_cost += self.penalty_weight * (max(0, self.T_start - At) + max(0, Dt - self.T_end))
        randomness = random.uniform(0.1, 1.0) #--- change this later 
        return total_cost + randomness

    def default_objective_function(self, params):
        C1, C2, C3, C4, Z, ETA, At, Dt, d = params

        cost_component1 = self.calculate_cost_component1([C1, Z])
        cost_component2 = self.calculate_cost_component2([C2, C3, C4, ETA, At, Dt, d])

        #total_cost = cost_component1 + cost_component2 + self.num_vessels * self.some_coefficient
        total_cost = cost_component1 + cost_component2 

        return total_cost

    def rosenbrock_function(self, x): # tune each of these functions 
        n = len(x)
        sum_term = 0
        for i in range(n - 1):
            sum_term += 100 * (x[i + 1] - x[i]**2)**2 + (x[i] - 1)**2
        transformed_value = sum_term**30 #----- change this ---
        return transformed_value
    
    def ackley_function(self, nests):  
            """  
            Calculate the value of the Ackley Function for given vectors.  
            Args:  
                nests (list or np.ndarray): A list of vectors or a single vector.  
            Returns:  
                float or list: The objective value(s) corresponding to each input vector.  
            """  
            results = []  
            
            # Check if nests is a 1D list or a 2D list of lists  
            if isinstance(nests, (list, np.ndarray)):  
                # If it's a 1D structure, wrap it in a list  
                if isinstance(nests[0], (int, float)):  # A single nest  
                    nests = [nests]  
            
            for x in nests:  
                if not isinstance(x, (list, np.ndarray)):  
                    raise ValueError("Each nest should be a list or numpy array.")  

                # Convert x into a NumPy array for calculations  
                x = np.array(x)  
                n = len(x)  
                sum_sq = np.sum(np.square(x))  
                sum_cos = np.sum(np.cos(2 * np.pi * x))  
                objective_value = (-20 * np.exp(-0.2 * np.sqrt(sum_sq / n))   
                                - np.exp(sum_cos / n) + 20 + np.e)  
                results.append(objective_value)  
            
            # Return a single value if only one nest was processed  
            if len(results) == 1:  
                return results[0]  
            transformed_value = results**30  #----- change this ---
            return transformed_value

    def rastrigin_function(self, x):  
            """  
            Calculate the value of the Rastrigin Function for a given vector x.  
            Args:  
                x (list or numpy.ndarray): A vector of n variables.  
            Returns:  
                float: The transformed objective value after applying the nonlinear transformation.  
            """  
            n = len(x)  
            x = np.array(x)  
            sum_term = np.sum(x**2 - 10 * np.cos(2 * np.pi * x))  
            value = 10 * n + sum_term
            # Apply a nonlinear transformation (e.g., exponential) to the final result  
            transformed_value = value**30 # #----- change this ---
            
            return transformed_value

    def griewank_function(self, x):  
            """  
            Calculate the value of the Griewank Function for a given vector x.  
            Args:  
                x (list or numpy.ndarray): A vector of n variables.  
            Returns:  
                float: The transformed objective value after applying the nonlinear transformation.  
            """  
            n = len(x)  
            x = np.array(x)  

            sum_sq = np.sum(x**2)  
            prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))  
            
            # Calculate the Griewank function value  
            griewank_value = 1 + sum_sq / 4000 - prod_cos  
            
            # Apply a nonlinear transformation (e.g., exponential) to the final result  
            transformed_value = griewank_value**30 #----- change this ---
            
            return transformed_value

    def schwefel_function(self, x):  
            """  
            Calculate the value of the Schwefel Function for a given vector x.  

            Args:  
                x (list or numpy.ndarray): A vector of n variables.  

            Returns:  
                float: The transformed objective value after applying the nonlinear transformation.  
            """  
            n = len(x)  
            x = np.array(x)  
            
            sum_term = np.sum(x * np.sin(np.sqrt(np.abs(x))))  
            schwefel_value = 418.9829 * n - sum_term  
            transformed_value = schwefel_value**30    #----- change this ---

            return transformed_value 

#objective_function_type = str(input("Enter the name of the objective function: "))

# Example usage
weights = [0.1, 0.2, 0.3, 0.4]
penalty_weight = 0.5
num_vessels = 3
some_coefficient = 0.05
objective_function_type = "calculate_cost_component2"
cost_calculator = CostCalculator(weights, penalty_weight, num_vessels, some_coefficient, objective_function_type)




