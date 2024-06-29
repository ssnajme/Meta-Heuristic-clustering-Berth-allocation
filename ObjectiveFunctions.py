import numpy as np

class CostCalculator:
    def __init__(self, weights, penalty_weight, num_vessels, some_coefficient, objective_function_type='default', T_start=0, T_end=10):
        self.weights = weights
        self.penalty_weight = penalty_weight
        self.num_vessels = num_vessels
        self.some_coefficient = some_coefficient
        self.objective_function_type = objective_function_type
        self.T_start = T_start
        self.T_end = T_end

    def objective_function(self, params):

        if self.objective_function_type == 'default':
            return self.default_objective_function(params)
        
        if self.objective_function_type == 'calculate_cost_component1':
            return self.calculate_cost_component1(params)
        
        if self.objective_function_type == 'calculate_cost_component2':
            return self.calculate_cost_component2(params)
        
        elif self.objective_function_type == 'rosenbrock':
            return self.rosenbrock_function(params)
        
        elif self.objective_function_type == 'ackley':
            return self.ackley_function(params)
        
        elif self.objective_function_type == 'rastrigin':
            return self.rastrigin_function(params)
        
        elif self.objective_function_type == 'griewank':
            return self.griewank_function(params)
        
        elif self.objective_function_type == 'schwefel':
            return self.schwefel_function(params)

        else:
            raise ValueError("Invalid objective function type")
        

    def calculate_cost_component1(self, params):
        C1, Z = params
        return self.weights[0] * C1 * (Z == 0)

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

    def default_objective_function(self, params):
        C1, C2, C3, C4, Z, ETA, At, Dt, d = params

        cost_component1 = self.calculate_cost_component1([C1, Z])
        cost_component2 = self.calculate_cost_component2([C2, C3, C4, ETA, At, Dt, d])

        total_cost = cost_component1 + cost_component2 + self.num_vessels * self.some_coefficient

        return total_cost

    def rosenbrock_function(self, x):
        n = len(x)
        sum_term = 0
        for i in range(n - 1):
            sum_term += 100 * (x[i + 1] - x[i]**2)**2 + (x[i] - 1)**2
        return sum_term
    
    def ackley_function(self, x):
        """
        Calculate the value of the Ackley Function for a given vector x.

        Args:
            x (list or numpy.ndarray): A vector of n variables.

        Returns:
            float: The objective value.
        """
        n = len(x)
        sum_sq = np.sum(np.square(x))
        sum_cos = np.sum(np.cos(2 * np.pi * x))
        return -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e

    def rastrigin_function(self, x):
        """
        Calculate the value of the Rastrigin Function for a given vector x.

        Args:
            x (list or numpy.ndarray): A vector of n variables.

        Returns:
            float: The objective value.
        """
        n = len(x)
        sum_term = np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
        return 10 * n + sum_term

    def griewank_function(self, x):
        """
        Calculate the value of the Griewank Function for a given vector x.

        Args:
            x (list or numpy.ndarray): A vector of n variables.

        Returns:
            float: The objective value.
        """
        n = len(x)
        sum_sq = np.sum(x**2)
        prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))
        return 1 + sum_sq / 4000 - prod_cos

    def schwefel_function(self, x):
        """
        Calculate the value of the Schwefel Function for a given vector x.

        Args:
            x (list or numpy.ndarray): A vector of n variables.

        Returns:
            float: The objective value.
        """
        n = len(x)
        sum_term = np.sum(x * np.sin(np.sqrt(np.abs(x))))
        return 418.9829 * n - sum_term

objective_function_type = str(input("Enter the name of the objective function: "))

# Example usage
weights = [0.1, 0.2, 0.3, 0.4]
penalty_weight = 0.5
num_vessels = 3
some_coefficient = 0.05

cost_calculator = CostCalculator(weights, penalty_weight, num_vessels, some_coefficient, objective_function_type=objective_function_type)


#params = np.array([1, 2, 3, 4, 0, 5, 6])
#total_cost = cost_calculator.objective_function(params)
#print(total_cost)



### Need to Include This 
""" 
def calculate_fitness(params, penalty_weight, weights, num_vessels, T_start, T_end, some_coefficient, preferred_quays, assigned_quays, x_it):
    C1, C2, C3, C4, Z, ETA, At, Dt, d = params
    # Add penalty terms for constraint violations
    if weights is None:
        weights = [1.1, 1.2, 1.3, 1.5] 
    cost = 0
    for k in range(len(Z)):
        cost += (
            weights[0] * C1[k] * Z[k]
            + weights[1] * C2[k] * np.minimum(ETA[k] - At[k], 0)
            + weights[2] * C3[k] * np.minimum(At[k] - ETA[k], 0)
            + weights[3] * C4[k] * np.minimum(Dt[k] - d[k], 0)
        )

        
        # include the constraints here to the OB function 
        ###########################
        if T: 
            if At[k]< T_start or Dt[k] > T_end:
                cost += penalty_weight * (max(0, T[0] - At[k]) + max(0, Dt[k] - T[1]))
        
        if num_vessels: 
            cost += num_vessels* some_coefficient  

    return cost

    
# do not include this in the objective function - only in the assignment  
################################################# 
def update_cost(preferred_quays, assigned_quays, x_it, penalty_weight):
    cost = 0
    for k, vessel_id in enumerate(preferred_quays.keys()):
        if assigned_quays and vessel_id in assigned_quays:
            if assigned_quays[vessel_id] != preferred_quays[vessel_id]:
            >> call the second objective function
                cost += x_it[k] * penalty_weight
    return cost
"""