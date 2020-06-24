import numpy as np
import itertools

###################### Decreasing the value of pheromones ###############################
#                                                                                       #
#  Description of Input Parameters:                                                     #
#  edge_table:            Edge Table                                                    #
#  beta:                  Parameter for updating pheromones                             #
#  gamma:                 Parameter for updating pheromones                             #
#  pop:                   Current Population                                            #
#                                                                                       #
#  Description of Output Parameters:                                                    #
#  edge_table:            Updated Edge Table                                            #
#                                                                                       #
#########################################################################################
def cap_phermones(edge_table,beta,gamma,pop):
	
	#print('Capping the value of Phermones')

	# Decrease the value of pheromones
	for indv in pop:

		indv = np.insert(indv,[0,indv.size],0)

		for start, end in zip(indv, indv[1:]):

			# Look for the rows where start location matches with the given start location
			start_loc_idx = np.argwhere(np.isin(edge_table[:,0],start))[:,0]

			# Among the selected rows, look for the rows in which end location matches the
			# required end location
			req_idx = np.argwhere(np.isin(edge_table[start_loc_idx,1],end))[:,0]

			edge_table[req_idx,4] = (1-beta)*edge_table[req_idx,4] + beta*gamma

	return edge_table
#---------------------------------------------------------------------------------------# 

###################### Increasing the value of pheromones ###############################
#                                                                                       #
#  Description of Input Parameters:                                                     #
#  edge_table:            Edge Table                                                    #
#  best_Indv:             Best Individual                                               #
#  bestFitness:           Fitness of the best individual                                #
#  alpha :                Parameter for updating pheromones                             #
#                                                                                       #
#  Description of Output Parameters:                                                    #
#  edge_table:            Updated Edge Table                                            #
#                                                                                       #
#########################################################################################
def increase_phermones(edge_table,best_Indv,bestFitness,alpha):
	
	#print('Increasing the value of Phermones')
	
	# Create a table with pairs of locations
	best_indv_ = np.insert(best_Indv,[0,best_Indv.size],0)

	for start, end in zip(best_indv_, best_indv_[1:]):
		
		# Look for the rows where start location matches with the given start location
		start_loc_idx = np.argwhere(np.isin(edge_table[:,0],start))[:,0]

		# Among the selected rows, look for the rows in which end location matches the
		# required end location
		req_idx = np.argwhere(np.isin(edge_table[start_loc_idx,1],end))[:,0]
		
		# Update the edge where both locations match
		edge_table[req_idx,4] = (1-alpha)*edge_table[req_idx,4] + alpha*bestFitness
	
	return edge_table
#---------------------------------------------------------------------------------------#