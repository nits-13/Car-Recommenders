# car_csp.py
class CarRecommendationCSP:
    def __init__(self, variables, domains, constraints):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.solutions = []  # Change to store multiple solutions with scores
        self.score = 0

    def solve(self):
        for row_values in zip(*self.domains.values()):
            assignment = dict(zip(self.variables, row_values))

            matching_parameters = sum(1 for var, value in assignment.items() if value in self.constraints.get(var, []) and var != 'car_name')   # Assign a score based on the number of matching parameters
            if matching_parameters > self.score:  
                self.solutions.append([assignment['car_name'],matching_parameters])
                self.score = matching_parameters 

        if self.solutions:
            return self.solutions
        else:
            return [None,None]

            
        """
        if self.solutions:
            max_score = max(self.solutions, key=lambda sol: sol['score'])['score']
            max_score_solutions = [sol for sol in self.solutions if sol['score'] == max_score]
            return max_score_solutions, max_score  # Return the cars with the maximum score and the maximum score
        else:
            return None, None
        """

    def select_unassigned_variable(self, assignment):
        unassigned_vars = [var for var in self.variables if var not in assignment]
        return min(unassigned_vars, key=lambda var: len(self.domains[var]))

    def order_domain_values(self, var, assignment):
        return self.domains[var]

    def is_consistent(self, var, value, assignment):
        matching_params = sum(1 for constraint_var in self.constraints.get(var, []) if constraint_var in assignment and assignment[constraint_var] == value)
        return matching_params > 0
