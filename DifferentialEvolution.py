import random
import numpy as np
from sklearn.utils import shuffle
F_P = 0.5
CR_P = 0.9
NP = 5
D = 5
Population = np.zeros((NP,D))
fitness = np.zeros(NP)
UB = 100
LB = -100

#generate random solutions between -100 and 100
def initialization(array, UB, LB,NP,D):
    array = np.random.uniform(LB, UB, size=(NP,D))
    return array


#mutation
def mutation(population, mutated_population, NP,D,F_P):
    random_vector1 = np.zeros((NP,5))
    for i in range(NP):
        random_vector1[i] = np.random.choice(np.arange(0, NP), replace=False, size=NP)
    for i in range(NP):
        mutated_population[i,:] = population[int(random_vector1[i,1]),:] + F_P * (population[int(random_vector1[i,2]),:]-population[int(random_vector1[i,3]),:])
    print(mutated_population)
    
     

Population = initialization(Population, UB, LB,NP,D)
mutated_population = np.zeros((NP,D))
mutation(Population, mutated_population, NP,D,F_P)

#crossover
def crossover(population,mutated_population, NP, D, CR_P):
    p = 0.5