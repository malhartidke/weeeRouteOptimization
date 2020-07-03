import numpy as np
from Distance_calc import distance_calc

######################## Creating Table for an Individual ###############################
#                                                                                       #
#  Description of Input Parameters:                                                     #
#  c_id_route:              Individual containing the sequence of customer IDs          #
#  tim_loc_data:            Input data containing                                       #
#                           1. Customer ID,                                             #
#                           2. Location of customer in terms of Latitude and Longitude  #
#                           3. Start and End Time intervals                             #
#                           4. Demand                                                   #
#                           (Modified after removing customers which are visited)       #
#                                                                                       #
#  Description of Output Parameters:                                                    #
#  vehicle_table:           A table with the following columns,                         #
#                           1. Customer IDs                                             #
#                           2. Start and End Times                                      # 
#                           3. Demand                                                   #
#                                                                                       #
#########################################################################################
def vehicle_dist(c_id_route,tim_loc_data):

	# Initialize empty table
	vehicle_table = np.zeros((c_id_route.size,6))
	
	idx = 0

	# Looping over the c_id_route
	for idx,c_id in enumerate(c_id_route):

		# If it is the depot, ignore it
		if c_id == 0:
			continue

		else:

			# Get the demand using Customer ID from given inout data
			row_copy = np.asarray(np.where(tim_loc_data[:,0] == c_id))[0][0]

			# Copy the obtained data in vehicle table
			vehicle_table[idx,:] = tim_loc_data[row_copy,(0,3,4,5,6,7)]


	# Delete all zeros in the vehicle table
	vehicle_table = vehicle_table[~np.all(vehicle_table == 0, axis=1)]
	
	return vehicle_table
#---------------------------------------------------------------------------------------#	

###################### Calculating Fitness of an Individual #############################
#                                                                                       #
#  Description of Input Parameters:                                                     #
#  indv:                    Individual containing the sequence of customer IDs          #
#  dat:                     Input data containing                                       #
#                           1. Customer ID,                                             #
#                           2. Location of customer in terms of Latitude and Longitude  #
#                           3. Start and End Time intervals                             #
#                           4. Demand                                                   #
#                           (Modified after removing customers which are visited)       #
#  travel_time_mat:         Matrix containing travel time for all edges                 #
#  dist_matx:               Distance matrix of all given customer nodes and depot,      #
#                           with depot being the first entry                            #
#  lat:                     Latitude of Depot                                           #
#  lon:                     Longitude of Depot                                          #
#  breakTimeStart:          Start of break time                                         #
#  breakTimeEnd:            End of break time                                           #
#  endTime:                 End of service time                                         #
#  Q:                       Capacity of each vehicle                                    #
#  min_dist:                Optimistic Distance                                         #
#  min_time:                Optimistic Time to complete service                         #
#  total_num_vehicles:      Total number of vehicles                                    #
#                                                                                       #
#  Description of Output Parameters:                                                    #
#  fit_ratio:               Fitness value of the individual                             #
#                                                                                       #
#########################################################################################
def fitness_func(indv,dat,travel_time_matx,dist_matx,lat,lon,breakTimeStart,breakTimeEnd,endTime,Q,min_dist,min_time,total_num_vehicles):

	# Create a table containing Customer IDs and demand of the customer in the order
	# of individual
	indv_table = vehicle_dist(indv,dat)

	# Calculate the distance for given individual                                                 
	act_dist, act_time, pen_time, idle_time, remaining = distance_calc(indv_table[:,:-1],Q,dist_matx,travel_time_matx,breakTimeStart,breakTimeEnd,endTime)
	
	if remaining:
		unattendedPenalty = np.sum(indv_table[remaining:,5]) / np.sum(indv_table[:,5])
	else:
		unattendedPenalty = 0

	if pen_time:
		pen_ratio = 1 / pen_time
	else:
		pen_ratio = 0.25
		
	# Calculate the fitness value
	fit_ratio = ((min_dist/total_num_vehicles)/act_dist) + ((min_time/total_num_vehicles)/act_time) + pen_ratio - unattendedPenalty

	return fit_ratio
#---------------------------------------------------------------------------------------#	