import random
import numpy as np
from sklearn.utils import shuffle
from eval import gtopx
from eval import print_results
from boundaries import Define_boundaries
F_P = 0.5 #mutation parameter
CR_P = 0.5 #crossover parameter
NP = 5 #Population size 
D = 5 #dimension
Population = np.zeros((NP,D)) #original population  
mutated_population = np.zeros((NP,D)) # mutated population
crossed_population = np.zeros((NP,D)) # crossed population
fitness = np.zeros(NP)

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
  
#crossover
def crossover(population,mutated_population, crossed_population, NP, D, CR_P):
    for i in range(NP):
        # Generating the random varibale delta
        dim = np.random.randint(0,D)
        for j in range(D):
           if (np.random.uniform()<CR_P or dim == j):
               crossed_population[i,j] = mutated_population[i,j]
           else :
               crossed_population[i,j] = population[i,j]

#boundaries handling
def boundaries_handling(UB, LB, population, D, num_benchmark, NP):
    for i in range(NP):
        # handling the integer variables for function 8 of gtopx by rounding them to the closest integer
        if (num_benchmark == 8) :
            for j in range(6,10):
               population[i,j] = round(population[i,j])
              
        # Bounding the violating variables to their upper bound
        population[i,:] = np.minimum(UB, population[i,:])
        # Bounding the violating variables to their lower bound
        population[i,:] = np.maximum(LB, population[i,:])
    return population


