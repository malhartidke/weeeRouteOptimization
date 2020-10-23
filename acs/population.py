import numpy as np
import math
from .fitness import fitness_func

################## Selecting Components to Form Individual ##############################
#                                                                                       #
#  Description of Input Parameters:                                                     #
#  edge_table:              Edge Table                                                  #
#  start_loc:               Customer ID of the current customer (0 in case of depot)    #
#  exploit_allow:           Parameter for allowance exploitation                        #
#                                                                                       #
#  Description of Output Parameters:                                                    #
#  end_location:            Customer ID corresponding to the end location               #
#                                                                                       #
#########################################################################################
def sel_comp(edge_table,start_loc,exploit_allow):
	
	# Generate a random number
	q = np.random.random()

	# Get the edges where the first column is start location 
	init_loc_idx = np.argwhere(edge_table[:,0] == start_loc).flatten()

	# Get the desirability
	desirability = edge_table[:,7]

	# Check if only one edge is remaining with the given start location
	if init_loc_idx.size == 1:

		# Return the corresponding end location and end function
		return(edge_table[init_loc_idx,1])

	# Else, check if generated random number is greater than exploit allowance
	if q > exploit_allow:

		# If it is, select two random components (end locations) from available locations 
		selected_indx = np.random.choice(np.arange(0,init_loc_idx.size),2,replace=False)

		# Get the best end location among selected two based on desirability
		max_idx = np.argmax(desirability[selected_indx])

		# Return the end location and end function
		return(edge_table[selected_indx[max_idx],1])
	else:

		# If the random value is less compared to exploit allowance, select the end location
		# with highest desirability and return it
		selected_indx = np.argmax(desirability[init_loc_idx])
		
		return(edge_table[selected_indx,1])
#---------------------------------------------------------------------------------------#

###################### Formation of Individual for ACS ##################################
#                                                                                       #
#  Description of Input Parameters:                                                     #
#  c_id:                    Customer ID                                                 #
#  n                        Number of vehicles for which to form individual             #
#  edge_table:              Edge Table                                                  #
#  exploit_allow:           Parameter for allowance exploitation                        #
#                                                                                       #
#  Description of Output Parameters:                                                    #
#  indv:                    Individual Formed                                           #
#                                                                                       #
#########################################################################################
def give_individual(c_id,n,edge_table,exploit_allow):
	
	# Get the number of customers
	num_cust = c_id.size

	# Divide the customers among the given number of vehicles equally
	ind_len = int(num_cust/n) + math.ceil((num_cust%n)/n)

	# Initialize the individual
	indv = np.zeros(ind_len)

	# Looping over the individual
	for i in range(ind_len):

		# Get the desirability of all edges from edge table
		desirability = edge_table[:,7].flatten()

		# Check if we are selecting from the depot
		if i == 0:

			# If yes, then get pass the start location as depot for selection 
			indv[i] = sel_comp(edge_table,0,exploit_allow)

		else:

			# Else, pass the current customer as in the individuals as start location 	
			indv[i] = sel_comp(edge_table,indv[i-1],exploit_allow)	

		# Since the current customer is selected, deleted the edges 
		# that lead to current customer 	
		edge_table = np.delete(edge_table,np.argwhere(edge_table[:,1] == indv[i]).flatten(),axis=0)
	
	return indv
#---------------------------------------------------------------------------------------#

###################### Formation of Population for ACS ##################################
#                                                                                       #
#  Description of Input Parameters:                                                     #
#  tim_loc_data:            Input data containing                                       #
#                           1. Customer ID,                                             #
#                           2. Location of customer in terms of Latitude and Longitude  #
#                           3. Start and End Time intervals                             #
#                           4. Demand                                                   #
#                           5. Value of the load at customer                            #
#                           6. Environmental value of load at the customer              # 
#                           (Modified after removing customers which are visited)       #
#  travel_time_mat:         Matrix containing travel time for all edges                 #  
#  c_id:                    Customer ID                                                 #
#  num_vehicles:            Number of vehicles for which to form population             #
#  edge_table:              Edge Table                                                  #
#  popsize_limit:           Maximum limit on the size of population                     #
#  dist_matrix:             Distance matrix of all given customer nodes and depot,      #
#                           with depot being the first entry                            #
#  breakTimeStart:          Start of break time                                         #
#  breakTimeEnd:            End of break time                                           #
#  endTime:                 End of service time                                         #
#  lat:                     Latitude of Depot                                           #
#  long:                    Longitude of Depot                                          #
#  Q:                       Capacity of each vehicle                                    #
#  init_min_dist:           Optimistic Distance                                         #
#  init_min_time:           Optimistic Time to complete service                         #
#  exploit_allow:           Parameter for allowance exploitation                        #
#  max_num_vehicles:        Total number of vehicles                                    #
#                                                                                       #
#  Description of Output Parameters:                                                    #
#  pop_:                    Population Formed                                           #
#  pop_fitness_:            Fitness of each individual in population                    #
#  Best_indv:               Best Individual obtained in the population                  #
#  best_fitnes_:            Fitness of the best individual obtained                     #
#                                                                                       #
#########################################################################################

def pop_form(tim_loc_data,travel_time_mat,c_id,num_vehicles,edge_table,popsize_limit,dist_matrix,lat,lon,breakTimeStart,breakTimeEnd,endTime,Q,init_min_dist,init_min_time,exploit_allow,max_num_vehicles):
	
	#print('Forming the population')
	
	##################################################################################
	# Initializing the parameters for iterations                                     #
	# popsize:              Current size of the population                           #
	# dummy_indv:           Initializing an individual using random dummy indivudual #
	##################################################################################
	popsize = 0
	dummy_indv = give_individual(c_id,num_vehicles,edge_table,exploit_allow)
	pop_ = np.zeros((popsize_limit,dummy_indv.size))
	pop_fitness_ = np.zeros(popsize_limit)

	# Initializing a random individual as best solution and calculating its fitness
	Best_indv = give_individual(c_id,num_vehicles,edge_table,exploit_allow)
	best_fitness_ = fitness_func(Best_indv,tim_loc_data,travel_time_mat,dist_matrix,lat,lon,breakTimeStart,breakTimeEnd,endTime,Q,init_min_dist,init_min_time,max_num_vehicles)

	# Looping until the population limit is reached
	while popsize < popsize_limit:

			# Get an individual
			indv = give_individual(c_id,num_vehicles,edge_table,exploit_allow)

			# Perform a local search on the obtained individual
			#indv = dmut(indv)
			
			# Calculate the fitness value of the individual
			fitness = fitness_func(indv,tim_loc_data,travel_time_mat,dist_matrix,lat,lon,breakTimeStart,breakTimeEnd,endTime,Q,init_min_dist,init_min_time,max_num_vehicles)

			# Check if the obtained indivudal has higher fitness than best individual
			if fitness >= best_fitness_:

				# If yes, replace the best individual with obtained individual and
				# also replace the fitness 
				Best_indv = indv
				best_fitness_ = fitness	

			# Add the individual to population and also add the corresponding fitness		
			pop_[popsize,:] = indv
			pop_fitness_[popsize] = fitness

			# Update the population size
			popsize = popsize + 1

	return pop_,pop_fitness_,Best_indv,best_fitness_
#---------------------------------------------------------------------------------------#	
