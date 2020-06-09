# Importing the required libraries
import numpy as np
import pandas as pd
from numpy import random
import math
import itertools
from sklearn.utils import shuffle
from Distance_calc import distance_calc, min_dist, form_edge_table

def sel_comp(edge_table,start_loc,exploit_allow):
	q = np.random.random()
	init_loc_idx = np.argwhere(edge_table[:,0] == start_loc).flatten()
	desirability = edge_table[:,4]
	if init_loc_idx.size == 1:
		return(edge_table[init_loc_idx,1])
	if q > exploit_allow:
		selected_indx = np.random.choice(np.arange(0,init_loc_idx.size),2,replace=False)
		max_idx = np.argmax(desirability[selected_indx])
		return(edge_table[selected_indx[max_idx],1])
	else:
		selected_indx = np.argmax(desirability[init_loc_idx])
		return(edge_table[selected_indx,1])

# This function is only used to return one set of parameters from the data when called
def give_individual(c_id,n,edge_table,exploit_allow):
	num_cust = c_id.size
	ind_len = int(num_cust/n) + math.ceil((num_cust%n)/n)
	indv = np.zeros(ind_len)
	for i in range(ind_len):
		desirability = edge_table[:,4].flatten()
		if i == 0:
			indv[i] = sel_comp(edge_table,0,exploit_allow)
		else:	
			indv[i] = sel_comp(edge_table,indv[i-1],exploit_allow)	
		edge_table = np.delete(edge_table,np.argwhere(edge_table[:,1] == indv[i]).flatten(),axis=0)
	return indv#,end_flag

def cap_phermones(edge_table,beta,gamma):
	#print('Capping the value of Phermones')
	edge_table[:,3] = (1-beta)*edge_table[:,3] + beta*gamma
	return edge_table 

def increase_phermones(edge_table,best_Indv,bestFitness,alpha):
	#print('Increasing the value of Phermones')
	best_indv_ = np.insert(best_Indv,[0,best_Indv.size],0)
	for idx,comp in enumerate(best_indv_):
		if idx == (best_indv_.size-1):
			break
		start_loc_idx = np.argwhere(np.isin(edge_table[:,0],best_indv_[idx]))[:,0]
		req_idx = np.argwhere(np.isin(edge_table[start_loc_idx,1],best_indv_[idx+1]))[:,0]
		edge_table[req_idx,3] = (1-alpha)*edge_table[req_idx,3] + alpha*bestFitness
	return edge_table

def pop_form(tim_loc_data,c_id,num_vehicles,edge_table,popsize_limit,dist_matrix,lat,lon,Q,init_min_dist,exploit_allow):
	#print('Forming the population')
	popsize = 0
	dummy_indv = give_individual(c_id,num_vehicles,edge_table,exploit_allow)
	pop_ = np.zeros((popsize_limit,dummy_indv.size))
	pop_fitness_ = np.zeros(popsize_limit)
	Best_indv = give_individual(c_id,num_vehicles,edge_table,exploit_allow)
	best_fitness_ = fitness_func(Best_indv,tim_loc_data,dist_matrix,lat,lon,Q,init_min_dist)
	while popsize < popsize_limit:
			indv = give_individual(c_id,num_vehicles,edge_table,exploit_allow)
			#indv = dmut(indv)
			fitness = fitness_func(indv,tim_loc_data,dist_matrix,lat,lon,Q,init_min_dist)
			#if popsize == 0:
			#	pop_ = np.zeros((popsize_limit,indv.size))
			#	pop_fitness_ = np.zeros(popsize_limit)
			if fitness >= best_fitness_:
				Best_indv = indv
				best_fitness_ = fitness		
			pop_[popsize,:] = indv
			pop_fitness_[popsize] = fitness
			popsize = popsize + 1
	return pop_,pop_fitness_,Best_indv,best_fitness_

def vehicle_dist(c_id_route,tim_loc_data):
	vehicle_table = np.zeros((c_id_route.size,2))
	idx = 0
	for idx,c_id in enumerate(c_id_route):
		if c_id == 0:
			continue
		else:
			row_copy = np.asarray(np.where(tim_loc_data[:,0] == c_id))[0][0]
			vehicle_table[idx,:] = tim_loc_data[row_copy,(0,3)]
	vehicle_table = vehicle_table[~np.all(vehicle_table == 0, axis=1)]
	return vehicle_table

# The fitness function which gives fitness value between 0 and 1
def fitness_func(indv,dat,dist_matx,lat,lon,Q,min_dist):
	indv_table = vehicle_dist(indv,dat)                                                 
	act_dist = distance_calc(indv_table,Q,dist_matx)            # Distance travelled by a vehicle for a given route
	dist_ratio = (min_dist/1)/act_dist                                                   # Calculating the fitness value
	return dist_ratio

def run_aco(tim_loc_data,dist_matrix,main_lat,main_long ,param_f,param_int,gen_num):                                                                               # Check if the module is imported
	no_of_runs = 1
	for i in range(no_of_runs):

		capacity = 25
		num_vehicles = 1
		popsize_limit = param_int[0]
		max_num_iter = param_int[1]

		alpha = param_f[0]				
		beta = param_f[1]
		delta = param_f[2]
		epsilon = param_f[3]
		gamma = param_f[4]
		exploit_allow = param_f[5]

		vehicle_num = 0
		num_iter = 0
		vech_iter = 0
		
		tim_loc_data_cpy = tim_loc_data 
		init_dist = distance_calc(tim_loc_data[:,(0,3)],capacity,dist_matrix)
		init_min_dist = min_dist(dist_matrix)
		print('\nInitial Distance is: %.2f'% init_dist)
		print('Initial Minimum Distance is: %.2f'% init_min_dist)
		print('Initial Fitness Value %.2f' % (init_min_dist/init_dist))

		edge_table = form_edge_table(dist_matrix)
		no_of_calls = tim_loc_data[:,0].size
		edge_table_size = edge_table[:,2].size
		
		components = edge_table[:,2]
		
		phermones = np.ones(edge_table_size)*gamma
		edge_table_cpy = np.append(edge_table, np.reshape(phermones,(edge_table_size,1)), axis=1)

		desirability = np.zeros((edge_table_size,1))
		edge_table_cpy = np.append(edge_table_cpy,desirability,axis=1)

		for vehicle_num in range(num_vehicles):
			#print('\nIterating...')
			if max_num_iter == 0:
				pop,pop_fitness,best_indv,best_fitness = pop_form(tim_loc_data,tim_loc_data_cpy[:,0],(num_vehicles-vehicle_num),edge_table_cpy,popsize_limit,dist_matrix,main_lat,main_long,capacity,init_min_dist,exploit_allow) 	
			else:
				for num_iter in range(max_num_iter):
					print('\nFor generation no. %d' %(gen_num))
					print('Forming solution for vehicle number %d' % (vehicle_num+1))	
					print('For iteration no. %d' %(num_iter+1))
					pop,pop_fitness,best_indv,best_fitness = pop_form(tim_loc_data,tim_loc_data_cpy[:,0],(num_vehicles-vehicle_num),edge_table_cpy,popsize_limit,dist_matrix,main_lat,main_long,capacity,init_min_dist,exploit_allow)			
					edge_table_cpy = cap_phermones(edge_table_cpy,beta,gamma)
					edge_table_cpy = increase_phermones(edge_table_cpy,best_indv,best_fitness,alpha)
					desirability = np.divide(np.power(edge_table_cpy[:,3],delta),np.power(edge_table_cpy[:,2],epsilon))
					edge_table_cpy[:,4] = desirability
					#if num_iter == 0 & vehicle_num == 0:
					#	np.savetxt('edge_table_init.csv',edge_table_cpy,delimiter=",")
				#np.savetxt('edge_table_'+str(vehicle_num)+'.csv',edge_table_cpy,delimiter=",")
			best_indv_idx_edge = np.argwhere(np.isin(edge_table_cpy[:,(0,1)],best_indv))[:,0]
			best_indv_idx_data = np.argwhere(np.isin(tim_loc_data_cpy[:,0],best_indv))[:,0]
			if vehicle_num == 0:
				best_sol = np.zeros((num_vehicles,best_indv.size))
			best_sol[vehicle_num,0:best_indv.size] = best_indv
			edge_table_cpy = np.delete(edge_table_cpy,best_indv_idx_edge,0)
			tim_loc_data_cpy = np.delete(tim_loc_data_cpy,best_indv_idx_data,0)
		#print('\nRoute taken by each vehicle is:')
		print(best_sol)
		#np.savetxt('best_sol.csv',best_sol,delimiter=",")
		best_fitnesses = np.apply_along_axis(fitness_func,1,best_sol,tim_loc_data,dist_matrix,main_lat,main_long,capacity,init_min_dist)
		best_sol_dist = np.zeros(num_vehicles)
		for vehc_route in best_sol:
			best_sol_table = vehicle_dist(vehc_route,tim_loc_data)
			best_sol_dist[vech_iter] = distance_calc(best_sol_table,capacity,dist_matrix)
			vech_iter = vech_iter + 1
		print(best_sol_dist)
		#np.savetxt('best_sol_dist.csv',best_sol_dist,delimiter=",")
		print('\nFinal Distance is: %f' %np.sum(best_sol_dist))
	return(np.sum(best_sol_dist))

#if __name__ == '__main__':
#	run_aco(np.array([ 0.56833394,  0.72499628,  0.81177863,  0.29492673,  0.83084323, 0.53222886]),np.array([54,16]),0)
	'''
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