import numpy as np
from acs.cvrp import run_aco

###################### Calculating Fitness of an Individual #############################
#                                                                                       #
#  Description of Input Parameters:                                                     #
#  design_vec:              Individual containing the set of parameters                 #
#  loc_dat:                 Input data containing                                       #
#                           1. Customer ID,                                             #
#                           2. Location of customer in terms of Latitude and Longitude  #
#                           3. Demand                                                   #
#                           (Modified after removing customers which are visited)       #
#  distances:               Distance matrix of all given customer nodes and depot,      #
#                           with depot being the first entry                            #
#  travel_time_mat:         Matrix containing travel time for all edges                 #
#  pos_lat:                 Latitude of Depot                                           #
#  pos_lon:                 Longitude of Depot                                          #
#                                                                                       #
#  Description of Output Parameters:                                                    #
#  total_val:               Fitness value of the individual                             #
#                                                                                       #
#########################################################################################
def fitness_func(design_vec,loc_data,distances,travel_time_mat,pos_lat,pos_long):

    # Intitializing the Variables
    min_final_penal = max_final_penal = 0
    max_pop_penalty = 0.1
    max_iter_penalty = 0.1
    
    prod = run_aco(loc_data,distances,travel_time_mat,pos_lat,pos_long,design_vec[:7],np.absolute(design_vec[7:9].astype(int)))                  # Calculate the output                         

    # Calculating the Hard Constraints
    for parameter in design_vec[:7]:
        
        # Checking if the value in set of parameters is below the minimum limits
        if parameter < 0:                        
            
            # Add the errors for all parameter
            min_final_penal = min_final_penal + abs(parameter - 0)

        else:

            # Checking if the value in set of parameters is above the maximum limits   
            if parameter > 1:          

                # Add the errors for all parameter
                max_final_penal = max_final_penal + abs(parameter - 1)

    # Calculating the total value of output
    total_val = -prod + min_final_penal + max_final_penal + (max_pop_penalty*design_vec[7]) + (max_iter_penalty*design_vec[8])

    # Return the score
    return total_val,
#---------------------------------------------------------------------------------------#