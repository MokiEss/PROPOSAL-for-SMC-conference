import DifferentialEvolution as DE
import numpy as np
from boundaries import Define_boundaries
from eval import gtopx

#the main optimize of DE
def optimize():
    #initialize parameters and necessary data
    num_function = 1 
    UB, LB, D, m = Define_boundaries(num_function) # get the upper bound, the lower bound, number of variables and number of constraints
    NP = D*14#Population size 
    maxNP = NP #max population size 
    minNP = 6  #min population size
    F_P = np.zeros(NP) + 0.5    #mutation parameter
    CR_P = np.zeros(NP) + 0.5 #crossover parameter
    Evaluation_number = 500000
    Population = np.zeros((NP,D)) #original population  
    mutated_population = np.zeros((NP,D)) # mutated population
    crossed_population = np.zeros((NP,D)) # crossed population
    eigen_crossed_population = np.zeros((NP,D))
    fitness_population = np.zeros(NP)
    constraints = np.zeros((NP,m))
    BestKnown_solution = np.zeros(D)
    fitness_best_solution = 10000000000
    it = 0
    #initialize the original population
    Population = DE.initialization(Population,UB, LB,NP, D)
    
    #boundary handling for mixed_integer problems
    Population = DE.boundaries_handling(UB, LB, Population, D, num_function, NP)
 
    #evaluation the population
    fitness_population, constraints = DE.Evaluate_population(num_function, Population, fitness_population,constraints, NP, 1,D,m)
    
    #get the best known solution of the initial population
    fitness_best_solution = np.amin(fitness_population)
    index_min = np.where(fitness_population == np.amin(fitness_population))
    BestKnown_solution = Population[index_min[0],:]
    
   
    #while stopping criteria is not met
    while (it<Evaluation_number):
      # 1- mutate
      F_P = np.random.uniform(size=NP)
      mutated_population = DE.mutation(Population, fitness_population,mutated_population,NP,D,F_P)
     
      # 2- cross
      CR_P = np.random.uniform(size=NP)
      eigen_population, eigen_mutated, eigen = DE.getEigenmatrix(Population,mutated_population)
      #crossed_population = DE.crossover(Population,mutated_population, crossed_population, NP, D, CR_P)
      eigen_crossed_population = DE.crossover(eigen_population,eigen_mutated, eigen_crossed_population, NP, D, CR_P)
      # return the eigen crossed population to the original coordinates
      crossed_population = np.matmul(eigen_crossed_population, eigen.T )
      # 3- boundary handling
      crossed_population = DE.boundaries_handling(UB, LB, crossed_population, D, num_function, NP)

      # 4- Evaluation of crossed_population
      fitness_crossed = np.zeros(NP) #empty array to store the fitness of crossed population
      constraints_crossed = np.zeros((NP,m)) #empty  2D array to store the constraints of crossed population
      
      fitness_crossed, constraints_crossed = DE.Evaluate_population(num_function, crossed_population, fitness_crossed,constraints_crossed ,NP, 1,D,m)
      
      # 5- Select the fitest solutions for the next generation
      Population, fitness_population = DE.Selection(crossed_population, Population, fitness_population, fitness_crossed, NP)

      # 6- Get rid of a fraction of the worst solution 
      Population, fitness_population, NP = DE.Reduce_population(Population, fitness_population, NP, maxNP, minNP, it, Evaluation_number)
      
      # 6- Get the best known solution and print it
      if (fitness_best_solution>np.amin(fitness_population)):
          fitness_best_solution = np.amin(fitness_population)
          index_min = np.where(fitness_population == np.amin(fitness_population))
          BestKnown_solution = Population[int(index_min[0]),:]
          
      it = it + NP
      print("the best solution so far is", fitness_best_solution, "current population size is", NP)
      
optimize()
