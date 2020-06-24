import numpy as np
from numpy import random
from deap import tools, algorithms
from .fitness import fitness_func
from scoop import futures

#################### Implmenting Simple Genetic Algorithm ##############################
# Description of Input Parameters:                                                     #
# call_data:       Input data containing,                                              #
#                  1. Customer ID,                                                     #
#                  2. Location of customer in terms of Latitude and Longitude          #
#                  3. Demand                                                           #
# dis_mat:         Distance Matrix of all given customer nodes and depot,              #
#                  with depot being the first entry                                    #
# travel_time_mat: Matrix containing travel time for all edges                         #
# lat:             Latitude of Depot                                                   #
# lon:             Longitude of Depot                                                  #
#                                                                                      #
# Description of Output Parameter:                                                     #
# HallOfFameBest:  Best set of input parameters for ACS                                #
# hallOfFameBest.fitness.values: Fitness value corresponding to HallOfFameBest         #
########################################################################################
def run_sga(call_data, dis_mat, travel_time_mat, lat, lon, toolbox):
                                 
    # Assign the Multiprocessor container
    toolbox.register("map", futures.map)
    
    # To get reproducible results
    #random.seed(1)              

    # CXPB : is the probability with which two individuals are crossed
    # MUTPB: is the probability for mutating an individual  
    CXPB, MUTPB = 0.8, 0.2

    # Assigning the evaluation function
    toolbox.register("evaluate",fitness_func, loc_data=call_data, distances=dis_mat, travel_time_mat=travel_time_mat, pos_lat=lat, pos_long=lon)
    
    # Creating a Hall of Fame 
    halloffame = tools.HallOfFame(10,similar=np.array_equal)
    
    # Create an initial population of individuals (where each individual is a set of parameters)
    pop = toolbox.population(n = 4)                                    
    
    # Apply Simple Genetic Algorithm
    pop = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=2, halloffame=halloffame)

    # Select the best individual from Hall of Fame
    hallOfFameBest = tools.selBest(halloffame, 1)[0]
    
    print("Evolution ended successfully...........")

    return hallOfFameBest, hallOfFameBest.fitness.values
#---------------------------------------------------------------------------------------#