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
#                  3. Start and End Time intervals                                     #
#                  4. Demand                                                           #
#                  5. Value of the load at customer                                    #
#                  6. Environmental value of load at the customer                      # 
# dis_mat:         Distance Matrix of all given customer nodes and depot,              #
#                  with depot being the first entry                                    #
# travel_time_mat: Matrix containing travel time for all edges                         #
# lat:             Latitude of Depot                                                   #
# lon:             Longitude of Depot                                                  #
# capacity:        Maximum volume capacity of each vehicle                             #
# max_num_vehicles:Maximum number of vehicles available                                #
# breakTimeStart:  Start of break time                                                 #
# breakTimeEnd:    End of break time                                                   #
# endTime:         End of service time                                                 #
# toolbox:         Deap toobox handle                                                  #
#                                                                                      #
# Description of Output Parameter:                                                     #
# HallOfFameBest:  Best set of input parameters for ACS                                #
# hallOfFameBest.fitness.values: Fitness value corresponding to HallOfFameBest         #
########################################################################################
def run_sga(call_data, dis_mat, travel_time_mat, lat, lon, capacity, max_num_vehicles, breakTimeStart, breakTimeEnd, endTime, locCount ,dfParam, toolbox):
                                 
    # Assign the Multiprocessor container
    toolbox.register("map", futures.map)
    
    # To get reproducible results
    #random.seed(1)              

    # CXPB : is the probability with which two individuals are crossed
    # MUTPB: is the probability for mutating an individual  
    CXPB, MUTPB = dfParam['CXTP'][0], dfParam['MUTPB'][0]

    # Assigning the evaluation function
    toolbox.register("evaluate",fitness_func, loc_data=call_data, distances=dis_mat, travel_time_mat=travel_time_mat, pos_lat=lat, pos_long=lon, capacity=capacity, max_num_vehicles=max_num_vehicles, breakTimeStart=breakTimeStart, breakTimeEnd=breakTimeEnd, endTime=endTime)
    
    # Creating a Hall of Fame 
    halloffame = tools.HallOfFame(dfParam['Hall of Fame Size'][0].astype(int),similar=np.array_equal)
    
    # Create an initial population of individuals (where each individual is a set of parameters)
    pop = toolbox.population(n = dfParam['SGA Max Population Size'][0].astype(int))                                    
    
    # Apply Simple Genetic Algorithm
    pop = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=dfParam['SGA Generations'][0].astype(int), halloffame=halloffame, verbose=False)

    # Select the best individual from Hall of Fame
    hallOfFameBest = tools.selBest(halloffame, 1)[0]
    
    print("\nEvolution ended successfully for "+str(locCount+1)+" location...........")

    return np.asarray(hallOfFameBest), hallOfFameBest.fitness.values
#---------------------------------------------------------------------------------------#