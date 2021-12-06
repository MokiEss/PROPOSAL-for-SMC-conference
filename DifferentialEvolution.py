import random
import numpy as np
F_P = 0.5
CR_P = 0.9
NP = 5
D = 5
Population = []
fitness = []
UB = 100
LB = -100
for i in range(0,NP) :
    Population.append([0 for c in range(0, D)])

#print data
def print_array(array, Nb_row):
    for i in range(0,Nb_row):
        print(array[i])

#generate random solutions between -100 and 100
def initialization(array, UB, LB,NP,D):
    array = np.random.uniform(LB, UB, size=(NP,D))
    return array


#mutation
def mutation(population, mutated_population, NP,D,F):



Population = initialization(Population, UB, LB,NP,D)
