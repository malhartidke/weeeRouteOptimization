# Importing the required libraries
from timeit import default_timer as timer
import sys
start_time = timer()
import numpy as np
from numpy import random
import math
import itertools
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from cvrp import run_aco
from scoop import futures
from Generate_Data import gen_data
from Distance_calc import dist_mat
import pandas as pd

# Define the Fitness function which returns score for the given set of parameters; Here we are maximizing the score; Score should be as near as possible to 1
# Adaptive Penalty Approach is used for Soft Constraints
# Static Penalty Approach is used for Hard Constraints
def fitness_func(design_vec,gen_num,loc_data,distances,pos_lat,pos_long):

    # Intitializing the Variables
    i = min_final_penal = max_final_penal = 0
    min_penal_para = 1000000                                
    max_penal_para = 1000000
    ref_val = 10000
    score = 0.0
    design_vec_ = np.asarray(design_vec[0])                                     # Converting the recieved 2D array to 1D array
    prod = run_aco(loc_data,distances,pos_lat,pos_long,design_vec_[0:6],np.absolute(design_vec_[6:8].astype(int)),gen_num)                  # Calculate the output                         

    # Calculating the Hard Constraints
    while i < len(design_vec_)-2:
        if (design_vec_[i] < 0):                       # Checking if the value in set of parameters is below the minimum limits 
            min_dist = abs(design_vec_[i] - 0)         # If it is, then calculate the error and square it so that it is always positive
            min_final_penal = min_final_penal + min_dist            # Add the errors for all parameter
             
        if (design_vec_[i] > 1):                       # Checking if the value in set of parameters is above the maximum limits   
            max_dist = abs(design_vec_[i] - 1)          # If it is, then calculate the error and square it so that it is always positive
            max_final_penal = max_final_penal + max_dist            # Add the errors for all parameter
        i+=1

    # Calculating the total value of output
    total_val = prod + (min_penal_para*min_final_penal) + (max_penal_para*max_final_penal)

    # Calulating the score
    score = (ref_val/total_val)
    # Return the score
    return score                                                   

# This function is only used to return one set of parameters from the data when called
def give_individual():
    param = np.zeros(8)
    param[0] = np.random.random()                
    param[1] = np.random.random()
    param[2] = np.random.random()
    param[3] = np.random.random()
    param[4] = np.random.random()
    param[5] = np.random.random()
    param[6] = np.random.randint(2,51,1)
    param[7] = np.random.randint(2,21,1)
    return(param)

# This function performs crossover between two set of parameters recieved
# Name of Cross-over Method: Simulated Binary Crossover
# This function is based on mathematical implementation
# Reference of mathematical implementation: "Simulated Binary Crossover for Continous Search Space" by K. Deb, R. Agrawal in Journal named "Complex Systems" in June 2000
def SimulatedBinary(ind1, ind2,q):

    '''
    Input Parameters:
    ind1, ind2: 2 Set of Parameters
    q: Cross-over Constant

    Output Parameters:
    ind1,ind2: 2 Set of Parameters after performing crossover
    '''
    
    i = 0
    ind1 = ind1[0]
    ind2 = ind2[0]
    size = len(ind1)
    while (i < len(ind1)):
        Pr1 = ind1[i]
        Pr2 = ind2[i]
        rand_no = np.random.random()
        if (rand_no < 0.5):
            alpha_prime = pow((2*rand_no),(1/(q+1)))
        elif (rand_no > 0.5):
            alpha_prime = pow((1/(2*(1-rand_no))),(1/(q+1)))
        else:
            alpha_prime = 1
        t1 = 0.5*(Pr1+Pr2)
        t2 = 0.5*alpha_prime*abs((Pr1-Pr2))
        ch1 = t1-t2
        ch2 = t1+t2
        ind1[i] = ch1
        ind2[i] = ch2
        i+=1
    return ind1, ind2

# This function performs mutation on a set of parameters recieved
# Name of Method: Polynomial Mutation
# This function is based on mathematical implementation
def mutPolynomial(in_mutant,delta_max,q):

    '''
    Input Parameters:
    in_mutant: Set of Parameters
    delta_max,q: Mutation Constants

    Output Parameters:
    in_mutant: Set of Parameters after performing mutation
    '''
    
    i = 0
    while (i < len(in_mutant)):
        rand_no = np.random.random()
        if (rand_no < 0.5):
            delta = pow((2*rand_no),(1/(q+1))) - 1
        elif (rand_no > 0.5):
            delta = 1 - pow((2*(1-rand_no)),(1/(q+1)))
        else:
            delta = 1
        in_mutant[i] += (delta*delta_max)
        i+=1
    return in_mutant

''' Creating a class 'FitnessMax' based on class 'base.Fitness'
    with the objective of maximizing single fitness value which
    is denoted by positive single value of weight'''
creator.create("FitnessMax", base.Fitness, weights=(1.0,))


''' Creating a class 'Individual' as a numpy array
    whose fitness is defined by the 'FitnessMax' class'''
creator.create("Individual", list, fitness=creator.FitnessMax)


toolbox = base.Toolbox()
toolbox.register("one_set_para",give_individual)                                             # Renaming the function 'give_individual' to 'one_set_para' which returns a set of parameter when called
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.one_set_para,1) # Individual is created by calling 'one_set_para' once
toolbox.register("population", tools.initRepeat, list, toolbox.individual)                   # Population is created by calling individual 'n' times; Here 'n' is specified when the population function is called
toolbox.register("evaluate",fitness_func)                                                    # Assigning the evaluation function
toolbox.register("mate", SimulatedBinary,q=5)                                                # Assigning the mating function and providing the mating parameters
toolbox.register("mutate", mutPolynomial,delta_max=0.1,q=5)                                  # Assigning the mutation function and providing the crossover parameters
toolbox.register("select", tools.selTournament, tournsize=2)                                 # Assigning the selection function and providing the tournament selection parameters


if __name__ == "__main__":                                                                   # Check if the module is imported
    toolbox.register("map", futures.map)
    
    lat = 26.9921975
    lon = 75.7840554
    
    file_location = 'time_loc_data.csv'                                                # Variable storing the location of file with file name
    try:
        df = pd.read_csv(file_location)                                 # Reading the CSV file
    except FileNotFoundError:                                                 # Error Handling    
        sys.exit()
    call_data = np.array(df)
    file_locationM = 'Dist_Matrix.csv'                                                # Variable storing the location of file with file name
    try:
        dfM = pd.read_csv(file_locationM)                                 # Reading the CSV file
    except FileNotFoundError:                                                 # Error Handling    
        sys.exit()
    dis_mat = np.array(dfM)
    
    #call_data = gen_data(lat,lon)
    #dis_mat = dist_mat(call_data,lat,lon)

    #random.seed(1)                                                        # To get reproducible results
    i = k = 0
    g = 0                                                                 # Variable keeping track of the number of generations
    min_fit_mean_inc = 0.1
    fit_mean_inc = min_fit_mean_inc + 1
    num_gen = 100
    fit_max = np.zeros(num_gen)
    fit_mean = np.zeros(num_gen)

    #CXPB : is the probability with which two individuals are crossed
    #MUTPB: is the probability for mutating an individual  
    CXPB, MUTPB = 0.8, 0.2
    
    pop = toolbox.population(n = 40)                                    # Create an initial population of 1000 individuals (where each individual is a set of parameters)
    print("\nStart of evolution...........")
    
    fitnesses = list(toolbox.map(toolbox.evaluate, pop, itertools.repeat(0,len(pop)),itertools.repeat(call_data,len(pop)),itertools.repeat(dis_mat,len(pop)),itertools.repeat(lat,len(pop)),itertools.repeat(lon,len(pop)))) # Evaluate the entire population
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit,
    
    print("Evaluated %i individuals" % len(pop))

    fits = [ind.fitness.values[0] for ind in pop]                         # Extracting all the fitnesses 

    while fit_mean_inc >= min_fit_mean_inc and g < num_gen:
    #while g < num_gen:                                                        # Begin the evolution
        print("\nGeneration %i" % g)
        offspring = toolbox.select(pop, len(pop))                         # Select the next generation individuals
        offspring = list(map(toolbox.clone, offspring))                   # Clone the selected individuals
    
        for child1, child2 in zip(offspring[::2], offspring[1::2]):       # Apply crossover and mutation on the offspring
            if random.random() < CXPB:                                    # Cross two individuals with probability CXPB
                toolbox.mate(child1, child2)
                del child1.fitness.values                                 # Fitness values of the children must be recalculated later
                del child2.fitness.values

        for mutant in offspring:
            in_mutant = mutant[0]                                         # Mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(in_mutant)
                del mutant.fitness.values
    
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid] # Evaluate the individuals with an invalid fitness
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, itertools.repeat(g,len(invalid_ind)),itertools.repeat(call_data,len(invalid_ind)),itertools.repeat(dis_mat,len(invalid_ind)),itertools.repeat(lat,len(invalid_ind)),itertools.repeat(lon,len(invalid_ind)))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit,

        print("Evaluated %i individuals" % len(invalid_ind))
        
        pop[:] = offspring                                                # The population is entirely replaced by the offspring
        fits = [ind.fitness.values[0] for ind in pop]                     # Gather all the fitnesses in one list and print the stats
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

        fit_max[g] = max(fits)
        fit_mean[g] = mean
        if g != 0:
            fit_mean_inc = (fit_mean[g]-fit_mean[g-1]/fit_mean[g-1])*100
        g = g + 1                                                         # A new generation
    
    print("Evolution ended successfully...........")
    
    best_ind = tools.selBest(pop, 1)[0]
    print("\nBest individual is %s, %s" % (best_ind, best_ind.fitness.values))

    best_individual =  best_ind[0]
    #Generating CSV Files
    print('\nGenerating CSV files..........')
    try:
        np.savetxt('Mean_fitness.csv',fit_max,delimiter=",")             # File containing the mean fitness values
        np.savetxt("best_coef.csv", best_individual, delimiter=",")       # File containing the best obtained values of parameters
    except PermissionError:
        print("The file is being used in another program or by another user. Please close the file and run this program again")
        print('CSV file could not be generated..........')
        print('The program has terminated')    
        sys.exit()
    print('CSV files generated succesfully...........')
    time_elapsed = (timer() - start_time)
    print('\nTime taken to run the code:',time_elapsed)