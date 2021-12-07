import random
import numpy as np
from sklearn.utils import shuffle
from eval import gtopx
from eval import print_results
F_P = 0.5 #mutation parameter
CR_P = 0.5 #crossover parameter
NP = 5 #Population size 
D = 5 #dimension
Population = np.zeros((NP,D)) #original population  
mutated_population = np.zeros((NP,D)) # mutated population
crossed_population = np.zeros((NP,D)) # crossed population
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

#test
#Population = initialization(Population, UB, LB,NP,D)
#print("original population")
#print(Population)
#mutation(Population, mutated_population, NP,D,F_P)
#print("mutated population")
#print(mutated_population)
#crossover(Population,mutated_population, crossed_population, NP, D, CR_P)
#print("crossed population")
#print(crossed_population)
benchmark = 1; print("\n Cassini1 ")
o  = 1; # number of objectives 
n  = 6; # number of variables 
ni = 0; # number of integer variables 
m  = 4; # number of constraints    
xl = [-1000.0,30.0,100.0,30.0,400.0,1000.0] # lower bounds
xu = [0.0,400.0,470.0,400.0,2000.0,6000.0]  # upper bounds
x  = [-789.759878, 158.29826, 449.38588, 54.7171393, 1024.686,  4552.799163] # best known solution
[f,g] = gtopx( benchmark, x,o,n,m ) # evaluate solution x
print_results(f,g)
