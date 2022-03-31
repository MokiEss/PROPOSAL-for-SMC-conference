import DifferentialEvolution as DE
import boundaries as bd
import numpy as np
from boundaries import Define_boundaries
from cec2017.functions import all_functions

# functions are defined for 10, 30, 50 and 100 dimensions
#the main optimize of DE
def optimize():
    #initialize parameters and necessary data
    num_function = 0
    D = 100
    UB, LB = bd.Define_boundaries_CEC2017(D)
    NP = D*14#Population size
    maxNP = NP #max population size 
    minNP = 6  #min population size
    mu_archive_size = 3
    mu_by_dimension =  np.zeros((mu_archive_size,D)) + (1/D)
    index_mu_by_dimension = 0
    F_P = np.zeros((NP,D)) + 0.5    #mutation parameter
    CR_P = np.zeros(NP) + 0.5 #crossover parameter
    Evaluation_number = 20000*D
    Population = np.zeros((NP,D)) #original population  
    mutated_population = np.zeros((NP,D)) # mutated population
    crossed_population = np.zeros((NP,D)) # crossed population
    eigen_crossed_population = np.zeros((NP,D))
    fitness_population = np.zeros(NP)
    #constraints = np.zeros((NP,m))
    BestKnown_solution = np.zeros(D)
    fitness_best_solution = 10000000000
    it = 0


    #initialize the multi criteria selection parameters
    number_of_operators = 2
    number_of_criteria = 3
    number_of_improved_solution = NP/2
    criteria_values = np.zeros((number_of_operators,number_of_criteria))
    for i in range(number_of_operators):
        for j in range(number_of_criteria):
            criteria_values[i,j] =  np.random.uniform()
            
    criteria_values[:,2] = number_of_improved_solution # 3rd criterion: set NP/2 as a number of improved solutions for both operators        
    operator_index = 0

    #initialize the original population
    Population = DE.initialization(Population,UB, LB,NP, D)
    
    #boundary handling for mixed_integer problems
    Population = DE.boundaries_handling_CEC2017(UB, LB, Population, D, num_function, NP)
 
    #evaluation the population

    fitness_population= DE.Evaluate_population_CEC2017(num_function, Population, fitness_population, NP,D)
    
    #get the best known solution of the initial population
    fitness_best_solution = np.amin(fitness_population)
    index_min = np.where(fitness_population == np.amin(fitness_population))
    BestKnown_solution = Population[index_min[0],:]
    
   
    #while stopping criteria is not met
    while (it<Evaluation_number):
      # 1- mutate
      #F_P = np.random.uniform(size=(NP,D))
      mutated_population,operator_index, criteria_values = DE.mutation(Population, fitness_population,mutated_population,NP,D,F_P, number_of_improved_solution,criteria_values, operator_index,BestKnown_solution)
     
      # 2- cross
      #CR_P = np.random.uniform(size=NP)
      #eigen_population, eigen_mutated, eigen = DE.getEigenmatrix(Population,mutated_population)
      crossed_population = DE.crossover(Population,mutated_population, crossed_population, NP, D, CR_P)
      #eigen_crossed_population = DE.crossover(eigen_population,eigen_mutated, eigen_crossed_population, NP, D, CR_P)
      # return the eigen crossed population to the original coordinates
      #crossed_population = np.matmul(eigen_crossed_population, eigen.T )
      # 3- boundary handling
      crossed_population = DE.boundaries_handling_CEC2017(UB, LB, crossed_population,D, num_function, NP)
    
      # 4- Evaluation of crossed_population
      fitness_crossed = np.zeros(NP) #empty array to store the fitness of crossed population
      #constraints_crossed = np.zeros((NP,m)) #empty  2D array to store the constraints of crossed population
      
      fitness_crossed= DE.Evaluate_population_CEC2017(num_function, crossed_population, fitness_crossed,NP,D)
     
      # 5- Select the fitest solutions for the next generation, determine the successfull F parameters and compute the fitness differences
      #Population, fitness_population, difference_fitness = DE.Selection(crossed_population, Population, fitness_population, fitness_crossed, NP, F_P,D)
      Population, fitness_population, difference_fitness,Nb_successful_parameters,SF_P,Successful_parents = DE.Selection(crossed_population, Population, fitness_population, fitness_crossed, NP, F_P,D)
      number_of_improved_solution = Nb_successful_parameters
      # 6- Generate new F values for each dimension and solution based on Cauchy distribution
      sF_P,mu_by_dimension,index_mu_by_dimension = DE.Generate_F_parameter(difference_fitness,D,F_P,NP,Nb_successful_parameters,SF_P,mu_by_dimension,index_mu_by_dimension,mu_archive_size,Successful_parents, Population)

      # 7- Get rid of a fraction of the worst solution 
      Population, fitness_population, NP = DE.Reduce_population(Population, fitness_population, NP, maxNP, minNP, it, Evaluation_number)
      
      # 8- Get the best known solution and print it
      if (fitness_best_solution>np.amin(fitness_population)):
          fitness_best_solution = np.amin(fitness_population)
          index_min = np.where(fitness_population == np.amin(fitness_population))
          BestKnown_solution = Population[int(index_min[0]),:]   
      it = it + NP
      print("the best solution so far is", fitness_best_solution, "current population size is", NP) 
      
       
optimize()

