import topsispy as tp
import numpy as np
number_of_operators = 2
number_of_criteria = 3
number_of_improved_solution = 10/2
criteria_values = np.zeros((number_of_operators,number_of_criteria))
criteria_values[:,0] = 0.5 # 1st criterion: set 0.5 as an average euclidean distance for both operators
criteria_values[:,1] = 0.5 # 2nd criterion: set 0.5 as an average standard deviation for both operators
criteria_values[:,2] = number_of_improved_solution # 3rd criterion: set NP/2 as a number of improved solutions for both operators
criteria_values[1,2] = 7
print(criteria_values)
print("---------")
w = [1, 1, 1]
sign = [1, 1, 1]
res = int(tp.topsis(criteria_values, w, sign)[0])
print(type(res))