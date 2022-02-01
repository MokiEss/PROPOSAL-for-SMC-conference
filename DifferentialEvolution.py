import random
import numpy as np
from sklearn.utils import shuffle
from eval import gtopx
from eval import print_results
from boundaries import Define_boundaries
import math as mt
from numpy import linalg as LA
#generate random solutions between -100 and 100
def initialization(population, UB, LB,NP,D):
    population = np.random.uniform(LB, UB, size=(NP,D))
    return population

#get the one of the best 10% solutions in the population
def Get_Pbest_solution(population, NB_p_best_solution,Sorted_index):
    index_best_solution = np.random.randint(0,NB_p_best_solution+1)
    return population[Sorted_index[index_best_solution],:]

#mutation
def mutation(population, fitness, mutated_population, NP,D,F_P):
    p = 0.1
    Sorted_index =  np.argsort(fitness)
    NB_p_best_solution = mt.floor(NP * p)
    random_vector1 = np.zeros((NP,NP))
    for i in range(NP):
        random_vector1[i] = np.random.choice(np.arange(0, NP), replace=False, size=NP)
    for i in range(NP):
        best_known_solution = Get_Pbest_solution(population, NB_p_best_solution,Sorted_index)
        mutated_population[i,:] = population[int(random_vector1[i,1]),:] + F_P[i,:] * (best_known_solution-population[int(random_vector1[i,3]),:]) + F_P[i,:] * (population[int(random_vector1[i,4]),:]-population[int(random_vector1[i,5]),:])
    return mutated_population
 

# compute covariance matrix and eigenvectors and transform the original population and the mutated one to the new eigen coordinates 
def getEigenmatrix(population, mutated_population):
    covariance_matrix = np.cov(population.T)
    w, eigen = LA.eigh(covariance_matrix)
    eigen_population = np.matmul(population, eigen)
    eigen_mutated = np.matmul(mutated_population, eigen)
    return eigen_population, eigen_mutated, eigen

#crossover
def crossover(population,mutated_population, crossed_population, NP, D, CR_P):
    for i in range(NP):
        # Generating the random varibale delta
        dim = np.random.randint(0,D)
        for j in range(D):
           # Check for donor vector or target vector
           if (np.random.uniform()<CR_P[i] or dim == j):
               # Accept variable from donor vector
               crossed_population[i,j] = mutated_population[i,j] 
           else :
               # Accept variable from target vector
               crossed_population[i,j] = population[i,j] 
    return crossed_population

#boundaries handling
def boundaries_handling(UB, LB, population, D, num_benchmark, NP):
    for i in range(NP):
        # handling the integer variables for function 8 of gtopx by rounding them to the closest integer
        if (num_benchmark == 8) :
            for j in range(6,10): # from 6 to 10 are the integer variables of function 8 of gtopx
               population[i,j] = round(population[i,j])
              
        # Bounding the violating variables to their upper bound
        population[i,:] = np.minimum(UB, population[i,:])
        # Bounding the violating variables to their lower bound
        population[i,:] = np.maximum(LB, population[i,:])
    return population

#evaluation
def Evaluate_population(function_num, population, fitness, constraints, NP, o,n,m):
    for i in range(NP):
        fitness_vector, c = gtopx( function_num, population[i],o,n,m )
        fitness[i] = fitness_vector[0]
        for j in range(m):
            constraints[i,j] = c[j]
    #return the fitness vector of the population and the values for constraints
    return fitness,constraints

#selection of the new generation
def Selection(crossed_population, population, fitness_vector, fitness_of_crossed, NP, F_P,D):
    SF_P = np.empty((0,D), dtype=float) #array to store succefull parmaters of F
    dif_fitness = np.zeros(NP)
    for i in range(NP):
        if fitness_of_crossed[i] < fitness_vector[i]: #greedy selection
            dif_fitness[i] = fitness_vector[i] - fitness_of_crossed[i]
            SF_P = np.vstack([SF_P,F_P[i,:]])
            population[i,:] = crossed_population[i,:]                    # Include the new solution in population
            fitness_vector[i] = fitness_of_crossed[i]
    # normalize differences of fitness to use it as weights later
    norm = np.linalg.norm(dif_fitness)
    difference_fitness = dif_fitness/norm
    return population, fitness_vector, difference_fitness, SF_P

#generate F parameter for each dimension to each solution
def Generate_F_parameter(SF_P,difference_fitness,D,F_P,NP):
    Nb_successful_parameters ,col = SF_P.shape
    weights = np.zeros(D)

    #compute the sum of fitness differences
    for i in range(Nb_successful_parameters):
         sum = sum + difference_fitness[i]

    #compute the weighted factors for each dimension
    for i in range(D):
        for j in range(Nb_successful_parameters):
            weights[i] = weights[i] + (SF_P[j,i]*difference_fitness[j])
        weights[i] = weights[i]/sum
    #generate F parameter for each dimension for each solution based on cauchy distribution
    for i in range(NP):
        for j in range(D):
            F_P[i,j] = weights[j] + 0.1*mt.tan(mt.pi*(np.random.uniform()-0.5))
            while F_P[i,j] <= 0:
                F_P[i,j] = weights[j] + 0.1*mt.tan(mt.pi*(np.random.uniform()-0.5))
            F_P[i,j] = min(F_P[i,j],0.99)
    return F_P

#reduction of the population
def Reduce_population(population, fitness, NP, maxNP, minNP, it, maxIT):
    plan_pop_size = round((((minNP - maxNP) / maxIT) * it) + maxNP)
    if NP > plan_pop_size:
        reduction_ind_num = NP - plan_pop_size
        if NP - reduction_ind_num < minNP:
            reduction_ind_num = NP - minNP
        indbest = np.argsort(fitness)
        for r in range(1,reduction_ind_num+1):
            worstind = indbest[NP-r]
            fitness = np.delete(fitness,worstind)
            population = np.delete(population,worstind,0)
        NP = NP - reduction_ind_num
    return population, fitness, NP


