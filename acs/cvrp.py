import math
import itertools
import numpy as np
from numpy import random
import pandas as pd
from sklearn.utils import shuffle
from Distance_calc import min_dist, form_edge_table, dist_mat, distance_calc
from .population import sel_comp, give_individual, pop_form
from .pheromones import cap_phermones, increase_phermones
from .fitness import fitness_func, vehicle_dist

###################### Implementation of ACS algorithm ##################################
#                                                                                       #
#  Description of Input Parameters:                                                     #
#  tim_loc_data:            Input data containing                                       #
#                           1. Customer ID,                                             #
#                           2. Location of customer in terms of Latitude and Longitude  #
#                           3. Start and End Time intervals                             #
#                           4. Demand                                                   #
#  dist_matrix:             Distance matrix of all given customer nodes and depot,      #
#                           with depot being the first entry                            #
#  travel_time_mat:         Matrix containing travel time for all edges                 #
#  main_lat:                Latitude of Depot                                           #
#  main_long:               Longitude of Depot                                          #
#  param_f:                 Set of input parameters for the ACS algorithm               #
#  param_int:               Set of iteration limits for the ACS algorithm               #
#                           1. Limit on Maximum population,                             #
#                           2. Limit on Maximum number of iterations                    #
#                                                                                       #
#  Description of Output Parameters:                                                    #
#  np.sum(best_sol_dist)    Sum of distance travelled by all the vehicles in the best   #
#                           solution                                                    #
#                                                                                       #
#########################################################################################

def run_aco(tim_loc_data,dist_matrix,travel_time_mat,main_lat,main_long ,param_f,param_int):

	##################################################################################
	# Retrieving the parameters for iterations in ACS                                #
	# popsize_limit:   Maximum limit on size of population                           #
	# max_num_iter:    Number of iterations to be done before reporting the solution #
	################################################################################## 
	popsize_limit = param_int[0]
	max_num_iter = param_int[1]

	##################################################################################
	# Defining problem specific parameters                                           #
	# capacity:       Maximum volume capacity of each vehicle                        #
	# num_vehicles:   Maximum number of vehicles available                           #
	##################################################################################
	capacity = 50
	num_vehicles = 5

    ##################################################################################
	# Retrieving the parameters for ACS                                              #
	# Refer the report for significance of each parameter                            #
	##################################################################################
	alpha = param_f[0]				
	beta = param_f[1]
	delta = param_f[2]
	epsilon = param_f[3]
	gamma = param_f[4]
	eta = param_f[5]
	exploit_allow = param_f[6]

	# Initializing the iterations parameters
	vehicle_num = 0
	num_iter = 0
	vech_iter = 0
	
	# Creating a copy of Input Data
	tim_loc_data_cpy = tim_loc_data

    ##################################################################################
	# Calculating and printing Initial Distances                                     #
	# init_dist:              The total distance travelled by each car if the        #
	#                         ascending order of customer ids is followed            #
	# init_min_dist:          Optimistic value of minimum total distance travelled   #
	# Initial Fistness Value: Fitness value corresponding to init_dist               #
	##################################################################################
	init_dist, init_total_time, init_penalty_time, init_idle_time = distance_calc(tim_loc_data[:,(0,3,4,5,6)],capacity,dist_matrix,travel_time_mat)
	init_min_dist = min_dist(dist_matrix)
	print('\nInitial Distance is: %.2f'% init_dist)
	print('\nInitial Total is: %.2f'% init_total_time)
	print('\nInitial Penalty is: %.2f'% init_penalty_time)
	print('\nInitial Idle is: %.2f'% init_idle_time)
	print('Initial Minimum Distance is: %.2f'% init_min_dist)

	# Forming the edge table
	edge_table = form_edge_table(dist_matrix, travel_time_mat)
	no_of_calls = tim_loc_data[:,0].size
	edge_table_size = edge_table[:,2].size
	
	# Initializing and adding Pheromones column in edge table
	phermones = np.ones(edge_table_size)*gamma
	edge_table_cpy = np.append(edge_table, np.reshape(phermones,(edge_table_size,1)), axis=1)

	# Initializing and adding Desirability column in edge table
	desirability = np.zeros((edge_table_size,1))
	edge_table_cpy = np.append(edge_table_cpy,desirability,axis=1)

	#Looping over the vehicles
	for vehicle_num in range(num_vehicles):

		#print('\nIterating...')
		
		# Check if the number of iterations are zero
		if max_num_iter == 0:

			# If they are, get the first selected population
			pop,pop_fitness,best_indv,best_fitness = pop_form(tim_loc_data,travel_time_mat,tim_loc_data_cpy[:,0],(num_vehicles-vehicle_num),edge_table_cpy,popsize_limit,dist_matrix,main_lat,main_long,capacity,init_min_dist,exploit_allow,num_vehicles) 	

		else:

			for num_iter in range(max_num_iter):

				print('Forming solution for vehicle number %d' % (vehicle_num+1))	
				print('For iteration no. %d' %(num_iter+1))

				# Form the population
				pop,pop_fitness,best_indv,best_fitness = pop_form(tim_loc_data,travel_time_mat,tim_loc_data_cpy[:,0],(num_vehicles-vehicle_num),edge_table_cpy,popsize_limit,dist_matrix,main_lat,main_long,capacity,init_min_dist,exploit_allow,num_vehicles)			
				
				# Decrease the pheromone value of all edges
				edge_table_cpy = cap_phermones(edge_table_cpy,beta,gamma,pop)

				# Increase the pheromone value of edges in best individual
				edge_table_cpy = increase_phermones(edge_table_cpy,best_indv,best_fitness,alpha)
				
				# Normalize each feature
				edge_table_cpy[:,2] = edge_table_cpy[:,2] / np.linalg.norm(edge_table_cpy[:,2])
				edge_table_cpy[:,3] = edge_table_cpy[:,3] / np.linalg.norm(edge_table_cpy[:,3])
				edge_table_cpy[:,4] = edge_table_cpy[:,4] / np.linalg.norm(edge_table_cpy[:,4])

				# Calculate new desirability
				desirability = np.divide(np.power(edge_table_cpy[:,4],delta),np.multiply(np.power(edge_table_cpy[:,3],eta),np.power(edge_table_cpy[:,2],epsilon)))
				
				# Update the desirability
				edge_table_cpy[:,4] = desirability

				# Block for saving the initial edge table or after each vehicle
				#if num_iter == 0 & vehicle_num == 0:
				#	np.savetxt('edge_table_init.csv',edge_table_cpy,delimiter=",")
				#np.savetxt('edge_table_'+str(vehicle_num)+'.csv',edge_table_cpy,delimiter=",")
		
		# Get the edges where start and end locations are in best individual
		best_indv_idx_edge = np.argwhere(np.isin(edge_table_cpy[:,(0,1)],best_indv))[:,0]
		# Delete the above obtained edges
		edge_table_cpy = np.delete(edge_table_cpy,best_indv_idx_edge,0)

		# Get the data corresponding to those locations
		best_indv_idx_data = np.argwhere(np.isin(tim_loc_data_cpy[:,0],best_indv))[:,0]
		# Delete the above obtained data
		tim_loc_data_cpy = np.delete(tim_loc_data_cpy,best_indv_idx_data,0)
		
		# If it is iteration for first vehicle, initialize empty solution 
		if vehicle_num == 0:
			best_sol = np.zeros((num_vehicles,best_indv.size))
			best_sol_fitness = np.zeros(num_vehicles)
		
		# Copy the best individual into the final best solution
		best_sol[vehicle_num,0:best_indv.size] = best_indv
		best_sol_fitness[vehicle_num] = best_fitness
	
	# Print and save the best solution
	print('\nRoute taken by each vehicle is:')
	print(best_sol)
	np.savetxt('best_sol.csv',best_sol,delimiter=",")

	# Calculate the distance covered by each vehicle
	best_sol_dist = np.zeros((num_vehicles,4))
	for vehc_route in best_sol:
		best_sol_table = vehicle_dist(vehc_route,tim_loc_data)
		best_sol_dist[vech_iter,0],best_sol_dist[vech_iter,1],best_sol_dist[vech_iter,2],best_sol_dist[vech_iter,3] = distance_calc(best_sol_table,capacity,dist_matrix,travel_time_mat)
		vech_iter = vech_iter + 1
	print(best_sol_dist)
	np.savetxt('best_sol_dist.csv',best_sol_dist,delimiter=",")

	# Sum the distance covered by each vehicle and return it
	print('\nFinal Distance is: %f' %np.sum(best_sol_dist[:,0]))
	print('\nTotal Time is: %f' %np.sum(best_sol_dist[:,1]))
	print('\nTotal Penalty Time is: %f' %np.sum(best_sol_dist[:,2]))
	print('\nTotal Idle Time is: %f' %np.sum(best_sol_dist[:,3]))
	return(np.average(best_sol_fitness,axis=0))
#---------------------------------------------------------------------------------------#


if __name__ == '__main__':
	
	lat = 26.9921975
	lon = 75.7840554

	file_location = 'time_loc_data.csv'                                                # Variable storing the location of file with file name
	try:
		df = pd.read_csv(file_location)                                 # Reading the CSV file
	except FileNotFoundError:                                                 # Error Handling    
		sys.exit()
	call_data = np.array(df)

	#dis_mat = dist_mat(call_data,lat,lon)
	
	file_locationM = 'Dist_Matrix.csv'                                                # Variable storing the location of file with file name
	try:
		dfM = pd.read_csv(file_locationM)                                 # Reading the CSV file
	except FileNotFoundError:                                                 # Error Handling    
		sys.exit()
	dist_mat = np.array(dfM)

	travel_time_mat = dist_mat/20
	
	param_f = np.array([0.2, 0.3, 0.08, 0.2, 0.3, 0.2, 0.8])
	param_int = np.array([250, 25])

	sol = run_aco(call_data,dist_mat,travel_time_mat,lat,lon,param_f,param_int)
	print(sol)

    #run_aco(np.array([ 0.56833394,  0.72499628,  0.81177863,  0.29492673,  0.83084323, 0.53222886]),np.array([54,16]),0)
	'''
	call_data = gen_data(lat,lon)
	capacity = 20
	train_file_location = 'time_loc_data.csv'                                                # Variable storing the location of file with file name
	try:
		df = pd.read_csv(train_file_location)                                 # Reading the CSV file
	except FileNotFoundError:                                                 # Error Handling    
		sys.exit()
	tim_loc_data = np.array(df)

	dist_file_location = 'Dist_Matrix.csv'                                                # Variable storing the location of file with file name
	try:
		mf = pd.read_csv(dist_file_location)                                 # Reading the CSV file
	except FileNotFoundError:                                                 # Error Handling
		sys.exit()
	dist_matrix = np.array(mf)
	vech_iter = 0
	best_sol = np.array([[1,4,7,10,13,16],[2,5,8,11,14,17],[3,6,9,12,15,18]])
	best_sol_dist = np.zeros(3)
	for vehc_route in best_sol:
			best_sol_table = vehicle_dist(vehc_route,tim_loc_data)
			best_sol_dist[vech_iter] = distance_calc(best_sol_table,capacity,dist_matrix)
			vech_iter = vech_iter + 1
	print(best_sol_dist)
	print('\nFinal Distance is: %f' %np.sum(best_sol_dist))	
	'''