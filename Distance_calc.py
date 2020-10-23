import numpy as np
import osmnx as ox
import networkx as nx

####################### Formation of Distance Matrix ####################################
#                                                                                       #
#  Description of Input Parameters:                                                     #
#  tim_loc_data:            Input data containing                                       #
#                           1. Customer ID,                                             #
#                           2. Location of customer in terms of Latitude and Longitude  #
#                           3. Demand                                                   #
#                           (Modified after removing customers which are visited)       #
#  main_lat:                Latitude of Depot                                           #
#  main_long:               Longitude of Depot                                          #
#                                                                                       #
#  Description of Output Parameters:                                                    #
#  matr:                    Distance Matrix                                             #
#                                                                                       #
#########################################################################################
def dist_mat(tim_loc_data,main_lat,main_long):
	
	print('\nCreating Distance Matrix......')
	
	# Get the pair of customer locations latitude and longitude
	service_pos_arr = tim_loc_data[:,1:3]
	
	# Loac the graph
	fp = 'VKI.graphml' 
	graph = ox.load_graphml(fp)

	# Get nodes closes to the required locations of depot and customers
	main_node = ox.get_nearest_node(graph,(main_lat,main_long),method='haversine')
	ids = ox.get_nearest_nodes(graph,service_pos_arr[:,1],service_pos_arr[:,0],method='balltree')
	ids = np.asarray(ids)
	
	# Create an empty distance matrix
	matr = np.zeros((np.size(tim_loc_data,0)+1,np.size(tim_loc_data,0)+1))

	# Looping over the matrix rowwise
	for x in range(np.size(matr,0)):

		# Looping over matrix columnwise
		for y in range(x+1):

			# If the column equal row, distance is zero
			if y == x:
				matr[x][y] = 0				
				continue

			else:

				# If the column for depot i.e. first column
				if y == 0:

					# Get the minimum distance using the graph
					try:
						matr[x][y] = nx.shortest_path_length(graph,source=main_node,target=ids[x-1],weight='length')#cost_calc(main_lat,main_long,service_pos_arr[x-1,0],service_pos_arr[x-1,1])
					
					# If path is not found, use default value
					except nx.exception.NetworkXNoPath:
						avgValue = 3500
						print('No Direct Path was found. Taking distance as average value of 3500 units')
						matr[x][y] = 3500					
				
				# If the column is not of depot
				else:

					# Get the minimum distance using the graph
					try:
						matr[x][y] = nx.shortest_path_length(graph,source=ids[x-1],target=ids[y-1],weight='length')#cost_calc(service_pos_arr[x-1,0],service_pos_arr[x-1,1],service_pos_arr[y-1,0],service_pos_arr[y-1,1])
					
					# If path is not found, use default value
					except nx.exception.NetworkXNoPath:
						avgValue = 3500
						print('No Direct Path was found. Taking distance as average value of 3500 units')
						matr[x][y] = 3500	

	# Get the indices of a upper triangular traingular matrix
	ul = np.triu_indices(np.size(matr,0))

	# Store the distance values in those indices, thus making the distance matrix symmetric
	matr[ul] = matr.T[ul]

	# Save the distance matrix and return it
	np.savetxt('Dist_Matrix.csv',matr,delimiter=",",header='Distance Matrix')
	print('Distance Matrix Created and Saved......')
	return matr
#---------------------------------------------------------------------------------------#	

########################## Implmenting the edge table ##################################
# Description of Input Parameters:                                                     #
# dist_matrix:     Distance Matrix of all given customer nodes and depot,              #
#                  with depot being the first entry                                    #
# travel_time_mat: Matrix containing travel time for all edges                         #
# valueData:       Array of values of each customer load                               #
# valueData:       Array of environmental values of each customer load                 #
#                                                                                      #
# Description of Output Parameter:                                                     #
# edge_table:      Edge table with rows containing edges and columns containing start  #
#                  nodes, end nodes and cost                                           #
########################################################################################

def form_edge_table(dist_matrix, travel_time_mat, valueData, envValueData):

    #################################################################################
	# Initializing the parameters                                                   #
	# n:           Number of rows in the distance matrix                            #
	# edge_table:  Initializing the edge table and fitting the distance matrix into #
	#              it                                                               #
	#################################################################################
	n = np.size(dist_matrix,0) 
	edge_table = np.zeros((dist_matrix.size,6))
	
	dist_matrix_flat = dist_matrix.flatten()
	travel_time_mat_flat = travel_time_mat.flatten()

	# Normalizing the Distances and Travel Time
	edge_table[:,2] = dist_matrix_flat/np.linalg.norm(dist_matrix_flat)
	edge_table[:,3] = travel_time_mat_flat/np.linalg.norm(travel_time_mat_flat)

    # Repeating 0 to n in the second column n times
	col_2 = np.arange(0,n)
	sec_col = np.repeat(np.reshape(col_2,(1,col_2.size)),n,axis=0)
	edge_table[:,1] = sec_col.flatten()
	
	valueData = np.insert(valueData,0,np.array([0]),axis=0)
	val_col = np.repeat(np.reshape(valueData,(1,valueData.size)),n,axis=0)
	val_col_flat = val_col.flatten()
	edge_table[:,4] = val_col_flat/np.linalg.norm(val_col_flat)

	envValueData = np.insert(envValueData,0,np.array([0]),axis=0)
	env_val_col = np.repeat(np.reshape(envValueData,(1,envValueData.size)),n,axis=0)
	env_val_col_flat = env_val_col.flatten()
	edge_table[:,5] = env_val_col_flat/np.linalg.norm(env_val_col_flat)

	# Repeating all elements 0 to n, n times each and putting them in first column
	edge_table[:,0] = np.repeat(col_2,n,axis=0)

	# Deleting the rows with zero distance
	edge_table = np.delete(edge_table, np.argwhere(edge_table[:,2] == 0).flatten(),axis=0)
	
    # Deleting the rows where depot is the end node 
	edge_table = np.delete(edge_table, np.argwhere(edge_table[:,1] == 0).flatten(),axis=0)
	
	return edge_table
#---------------------------------------------------------------------------------------#

############### Implementation of the Optimistic Distance Calculator ###################
# Description of Input Parameters:                                                     #
# matr:           Distance Matrix of all given customer nodes and depot,               #
#                 with depot being the first entry                                     #
#                                                                                      #
# Description of Output Parameter:                                                     #
# min_dist:       Optimistic Minimum distance travelled obtained by adding minimum     # 
#                 distance from each column                                            #
########################################################################################

def min_dist(matr):
	
	min_dist = 0

	# Looping through the columns of distance matrix
	for idx in range(np.size(matr,1)-1):
		
		col = matr[:,idx]
		
		# Getting the minimum distance from that column and adding it
		min_dist = min_dist + col[np.where(col==np.min(col[np.nonzero(col)]))[0][0]]

	return min_dist
#---------------------------------------------------------------------------------------#

######### Implementation of Distance Calculation in given order of Customer IDs #########
#                                                                                       #
# Description of Input Parameters:                                                      #
# customer_info:   Input data containing                                                #
#                  1. Customer ID                                                       #
#                  2. Demand of the customer                                            #
# capacity:        Capacity of vehicles                                                 #
# dist_mat:        Distance matrix of all given customer nodes and depot,               #
#                  with depot being the first entry                                     #
# travel_time_mat: Matrix containing travel time for all edges                          #
# breakTimeStart:  Start of break time                                                  #
# breakTimeEnd:    End of break time                                                    #
# endTime:         End of service time                                                  #
# idx:             Index from where customers remain unattended                         #
#                                                                                       #
# Description of Output Parameters:                                                     #
# total_dist:      Total distance travelled by all the vehicles                         #
# total_time:      Total time taken for travel by all the vehicles                      #
# penalty_time:    Total late time for all vehicles                                     #
# idle_time:       Total idle time for all the vehicles                                 #                                                            #
######################################################################################### 

def distance_calc(customer_info,capacity,dist_mat,travel_time_mat,breakTimeStart,breakTimeEnd,endTime):
	
	################################################################################## 
	# Retrieving the customer data                                                   #
	# customer_load:   Load of the particular customer                               #
	# customer_id:     Customer ID                                                   #
	##################################################################################
	customer_id = customer_info[:,0]
	customer_start_time = customer_info[:,1]
	customer_end_time = customer_info[:,2]
	customer_load_min = customer_info[:,3]
	customer_load_max = customer_info[:,4]

    ##################################################################################
	# Initializing the parameters for iterations                                     #
	# reached_home:    Value is 1 if the vehicle is at depot and 0 otherwise         #
	# total_load:      Load in the vehicle at any instance                           #
	##################################################################################
	reached_home = 1
	total_dist = 0 
	total_load = 0
	total_time = 0
	penalty_time = 0
	idle_time = 0

	# Looping over customers
	for idx, c_id in enumerate(customer_id):

		# Checking of overtime has occured
		overTime = total_time - endTime
		if overTime >= 0:
			#print('Overtime has occured')
			return total_dist,total_time,penalty_time,idle_time,idx

		# Checking if it is break time
		breakExtra = total_time - breakTimeStart
		if breakExtra >= 0:
			total_time = breakTimeEnd + breakExtra

		# Check if the vehicle has reached the depot or another customer
		if reached_home == 1:

			# If the vehicle has reached depot, add the distance and time between  
			# previous customer and depot
			total_dist = total_dist + dist_mat[int(customer_id[idx])][0]
			total_time = total_time + travel_time_mat[int(customer_id[idx])][0]
		
		else:

			# Else add the distance between previous customer and the customer at which
			# the vehicle has arrived
			total_dist = total_dist + dist_mat[int(customer_id[idx-1])][int(customer_id[idx])]
			total_time = total_time + travel_time_mat[int(customer_id[idx-1])][int(customer_id[idx])]

			late_time = total_time - customer_end_time[idx]
			if late_time >= 0:
				penalty_time = penalty_time + late_time
			else:
				early_time = customer_start_time[idx] - total_time
				if early_time >= 0:
					idle_time = idle_time + early_time
		
		# Add the load recieved at the current customer
		total_load = total_load + np.random.randint(customer_load_min[idx],customer_load_max[idx]+1)

		# Check if current customer is the last customer 
		if idx == (customer_id.size-1):

			# If it is the last customer, add the distance between this customer
			# and depot and end the loop 
			total_dist = total_dist + dist_mat[int(customer_id[idx])][0]
			total_time = total_time + travel_time_mat[int(customer_id[idx])][0]
			break

		else:

			# Check if the load at next customer can be picked up without violating the
			# capacity constraint
			if (total_load + customer_load_max[idx+1]) <= capacity:				
			
				# If yes, take the vehicle back to next customer	
				reached_home = 0
				continue
			
			else:

				# Else, take the vehicle back to depot and add the distance between 
				# current customer and depot and unload the vehicle
				total_dist = total_dist + dist_mat[int(customer_id[idx])][0]
				total_time = total_time + travel_time_mat[int(customer_id[idx])][0]

				reached_home = 1
				total_load = 0

	return total_dist,total_time,penalty_time,idle_time,0
#---------------------------------------------------------------------------------------#

## File Test Code ##
'''
if __name__ == '__main__':
	import pandas as pd
	train_file_location = 'time_loc_data.csv'
	df = pd.read_csv(train_file_location)
	tim_loc_data = np.array(df)
	main_long = 75.7840554
	main_lat = 26.9921975
	a = dist_mat(tim_loc_data,main_lat,main_long)
'''