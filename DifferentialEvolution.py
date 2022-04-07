import random
import numpy as np

from sklearn.utils import shuffle
from eval import gtopx
from eval import print_results
from boundaries import Define_boundaries
import math as mt
from numpy import linalg as LA
import topsispy as tp
from cec2017.functions import all_functions

#generate random solutions between -100 and 100
def initialization(population, UB, LB,NP,D):
    population = np.random.uniform(LB, UB, size=(NP,D))
    return population

#get the one of the best 10% solutions in the population
def Get_Pbest_solution(population, NB_p_best_solution,Sorted_index,NP):
    index_best_solution = np.random.randint(0,NB_p_best_solution+1)
    index_worst_solution = (NP-1) - index_best_solution
    return population[Sorted_index[index_best_solution],:], population[Sorted_index[index_worst_solution],:]


#compute the average euclidean distance between individuals and the best solution so far
def Average_euclidean_distance(population, best_solution, NP):
    total_distance = 0
    for i in range(NP):
        total_distance = mt.sqrt(np.sum((population[i,:]-best_solution)**2))+total_distance
    return total_distance/NP

#compute the average standard deviation of the population as an exploration criteria
def Average_standard_deviation(population,D):
    total_standard_deviation = 0
    for i in range(D):
        total_standard_deviation = total_standard_deviation + np.std(population[:,i])
    return total_standard_deviation/D

#mutation
def mutation(population, fitness, mutated_population, NP,D,F_P, number_of_improved_solution,criteria_values, operator_index, best_solution):

    w = [1, 1, 1]
    sign = [1, 1, 1]
   
    operator_index = int(tp.topsis(criteria_values, w, sign)[0])


    #print("the strategy to be used is :",operator_index)
    # end multi-criteria selection
    p = 0.1
    Sorted_index =  np.argsort(fitness)
    NB_p_best_solution = mt.floor(NP * p)
    random_vector1 = np.zeros((NP,NP))
    for i in range(NP):
        random_vector1[i] = np.random.choice(np.arange(0, NP), replace=False, size=NP)
    for i in range(NP):
        best_known_solution, worst_known_solution  = Get_Pbest_solution(population, NB_p_best_solution,Sorted_index,NP)
        if operator_index==1 :
            for j in range(D):
                mutated_population[i,j] = population[i,j] + F_P[i,j] * (best_known_solution[j] - population[int(random_vector1[i,1]),j]) + F_P[i,j] * (population[int(random_vector1[i,2]),j]-worst_known_solution[j])
        else :
            for j in range(D):
                #mutated_population[i, j] = F_P[i,j] * population[i,j]  + (best_known_solution[j] - population[int(random_vector1[i,1]),j])
                mutated_population[i,j] = population[int(random_vector1[i,2]),j] + F_P[i,j] * (population[i,j]-population[int(random_vector1[i,3]),j]) + F_P[i,j] * (population[int(random_vector1[i,0]),j]-population[int(random_vector1[i,1]),j])

                # mutated_population[i,j] = F_P[i,j] * population[i,j] + (best_known_solution[j] - population[int(random_vector1[i,1]),j])
     
    # begin multi_criteria selection to choose the mutation strategy
    # 1st criteria : average euclidean distance to the best solution so far
    criteria_values[operator_index,0] = Average_euclidean_distance(population, best_solution, NP)
    # 2nd criteria : average standard deviation of the population
    criteria_values[operator_index,1] = Average_standard_deviation(population,D)
    # 3rd criteria : number of improved solutions
    criteria_values[operator_index,2] = number_of_improved_solution
    # prepare weights for criteria            
    return mutated_population, operator_index, criteria_values
 

# compute covariance matrix and eigenvectors and transform the original population and the mutated one to the new eigen coordinates 
def getEigenmatrix(population, mutated_population,D):
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
           if (np.random.uniform()<CR_P[i,j] ):
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

#boundaries handling for cec 2017
def boundaries_handling_CEC2017(UB, LB, population, D, num_benchmark, NP):
    for i in range(NP):
        # Bounding the violating variables to their upper bound
        population[i, :] = np.minimum(UB, population[i, :])
        # Bounding the violating variables to their lower bound
        population[i, :] = np.maximum(LB, population[i, :])
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


#evaluation
def Evaluate_population_CEC2019(function_num, population, fitness, NP, D, bench):

    for i in range(NP):
        fitness[i] = bench.eval(population[i,:])

    #return the fitness vector of the population and the values for constraints
    return fitness

def Evaluate_population_CEC2017(function_num, population, fitness, NP, D):

    for i in range(NP):
        fitness[i] = all_functions[function_num](population[i,:])

    #return the fitness vector of the population and the values for constraints
    return fitness

#selection of the new generation
def Selection(crossed_population, population, fitness_vector, fitness_of_crossed, NP, F_P,CR_P,D):
    SF_P = np.empty((0,D), dtype=float) #array to store successfull parmaters of F
    SCR_P = np.empty((0,D), dtype=float) #array to store successfull parmaters of CR
    Successful_parents = np.empty((0,D), dtype=float) #array to store successfull parents
    dif_fitness = np.zeros(NP)
    difference_fitness = np.zeros(NP)
    Nb_successful_parameters = 0
    for i in range(NP):
        if fitness_of_crossed[i] < fitness_vector[i]: #greedy selection
            dif_fitness[i] = fitness_vector[i] - fitness_of_crossed[i]
            SF_P = np.vstack([SF_P,F_P[i,:]])
            SCR_P = np.vstack([SCR_P ,CR_P[i,:]])
            Successful_parents = np.vstack([Successful_parents,crossed_population[i,:]-population[i,:]])
            population[i,:] = crossed_population[i,:]                    # Include the new solution in population
            fitness_vector[i] = fitness_of_crossed[i]
            Nb_successful_parameters = Nb_successful_parameters + 1
    # normalize differences of fitness to use it as weights later
    if Nb_successful_parameters>0:
        norm = np.linalg.norm(dif_fitness)
        difference_fitness = dif_fitness/norm
    return population, fitness_vector, difference_fitness,Nb_successful_parameters,SF_P,SCR_P,Successful_parents

#generate F parameter for each dimension to each solution
def Generate_F_CR_parameter(difference_fitness,D,F_P,CR_P,NP,Nb_successful_parameters,SF_P,SCR_P,mu_by_dimension,mu_by_dimensionCR,index_mu_by_dimension,mu_archive_size,Successful_parents, population):
    norm = np.zeros(D)
    if index_mu_by_dimension>mu_archive_size-1:
       index_mu_by_dimension = 0
    
    #compute the contribution of each dimnension in improving the fitness value
    if Nb_successful_parameters > 0 :
        # 1- normalize the Successful_parents to compute the contribution of each dimension
        for i in range(D):
            norm[i] = np.max(population[:,i]) - np.min(population[:,i])
            Successful_parents[:,i] = Successful_parents[:,i]/norm[i]
        
        for i in range(D):
            sum_by_dimension_2 = 0
            sum_by_dimension = 0
            sum_by_dimension_2_CR = 0
            sum_by_dimension_CR = 0
            for j in range(Nb_successful_parameters):
                sum_contribute = 0
                for k in range(D):
                    sum_contribute = sum_contribute + Successful_parents[j,k]
                sum_by_dimension_2 = sum_by_dimension_2 + ((Successful_parents[j,i]/sum_contribute)*SF_P[j,i])**3
                sum_by_dimension = sum_by_dimension + ((Successful_parents[j,i]/sum_contribute)*SF_P[j,i])**2
                sum_by_dimension_2_CR = sum_by_dimension_2_CR + ((Successful_parents[j,i]/sum_contribute)*SCR_P[j,i])**3
                sum_by_dimension_CR = sum_by_dimension_CR + ((Successful_parents[j,i]/sum_contribute)*SCR_P[j,i])**2
        #avoid nan values for mu_by_dimension
            if mt.isnan(sum_by_dimension_2/sum_by_dimension) :
                mu_by_dimension[index_mu_by_dimension,i] = 1/D
                mu_by_dimensionCR[index_mu_by_dimension,i] = 1/D
            else :
                mu_by_dimension[index_mu_by_dimension,i] = (sum_by_dimension_2/sum_by_dimension)
                mu_by_dimensionCR[index_mu_by_dimension, i] = (sum_by_dimension_2_CR/sum_by_dimension_CR)
        #normalize mu_by_dimension so that the sum of columns of a given row gives 1
      
        index_mu_by_dimension = index_mu_by_dimension + 1 
    

    #randomly choose a mu value from the archive mu_by_dimension
    
    mu_index = np.random.randint(0,mu_archive_size)
    #generate F parameter for each dimension for each solution based on cauchy distribution
   
    for i in range(NP):
        for j in range(D):
            F_P[i,j] = mu_by_dimension[mu_index,j] + 0.1*mt.tan(mt.pi*(np.random.uniform()-0.5))
            CR_P[i,j] = np.random.normal(mu_by_dimensionCR[mu_index,j],0.1)
            #while F_P[i,j] <= 0:
                #F_P[i,j] = mu_by_dimension[mu_index,j] + 0.1*mt.tan(mt.pi*(np.random.uniform()-0.5))
            F_P[i, j] = max(F_P[i, j], 0.1)
            CR_P[i, j] = max(CR_P[i, j], 0.1)
            F_P[i,j] = min(F_P[i,j],0.99) #the value of F should be between 0 and 1
            CR_P[i, j] = min(CR_P[i, j], 0.99) #the value of CR should be between 0 and 1
    #print(F_P)        
    return F_P,mu_by_dimension,mu_by_dimensionCR,index_mu_by_dimension

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


