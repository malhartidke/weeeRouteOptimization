import osmnx as ox
import geopandas as gpd
import numpy as np
import networkx as nx
import itertools
import scipy.stats as stats

############################# Generation of Data ########################################
#                                                                                       #
#  Description of Input Parameters:                                                     #
#  lat:                     Latitude of Depot                                           #
#  lon:                     Longitude of Depot                                          #
#                                                                                       #
#  Description of Output Parameters:                                                    #
#  tim_loc_data:            Input data containing                                       #
#                           1. Customer ID,                                             #
#                           2. Location of customer in terms of Latitude and Longitude  #
#                           3. Demand                                                   #
#                                                                                       #
#########################################################################################
def gen_data(lat,lon):

	print('\nGenerating CSV file..........')
	
	# Load the graph
	fp = 'VKI.graphml'
	graph = ox.load_graphml(fp)

	# Remove the isolated nodes
	graph.remove_nodes_from(nx.isolates(graph))

	#fig, ax = ox.plot_graph(graph, node_size=30, node_color='#66cc66')
	
	# Extracts nodes and edges from the graph
	nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
	
	# Get the node closest to Depot
	main_node = ox.get_nearest_node(graph,(lat,lon),method='haversine')
	
	# Get all nodes with potential to be customer nodes
	possibleNodes = np.asarray(nodes)[:,2]

	# Select only the nodes which have connectivity to the depot
	actualNodes = list(map(nx.has_path,itertools.repeat(graph,len(possibleNodes)),itertools.repeat(main_node,len(possibleNodes)),possibleNodes))
	actualIdx = list(itertools.compress(range(len(actualNodes)), actualNodes))
	finalNodes = nodes.iloc[actualIdx,:]

	# Number of calls recieved in a day are taken at random from a normal distribution with defined mean and standard deviation
	mu_c, sigma_c = 120,10 
	no_of_calls = int(np.random.normal(mu_c,sigma_c))
	randomChoiceArr = np.random.choice(finalNodes.shape[0],no_of_calls,replace=False)
	
	# Create a table with locations of customers in terms of latitude and longitude
	service_pos_arr = np.asarray(finalNodes.iloc[randomChoiceArr,:])[:,(0,1)]

	# The demand of each customer is taken at random from a normal distribution with defined mean and standard deviation
	mu_l, sigma_l = 10,1
	load_deviation = 1
	customer_load = np.random.normal(mu_l,sigma_l,no_of_calls).astype(int)
	load_uncertainity_add = np.add(customer_load,load_deviation)
	load_uncertainity_sub = np.subtract(customer_load,load_deviation)
	customer_load = np.append(np.reshape(load_uncertainity_sub,(-1,1)),np.reshape(load_uncertainity_add,(-1,1)),axis=1)

	# Get the number of working hours including the break hour 
	no_of_working_hrs = 9
	# Get the start time, break time and duration of break time in 24 hrs format
	start_time = 9
	break_time = 12.5
	break_time_duration = 1.5

	# Create all possible time minutes which can be booked 
	tim_arr_before = np.arange(0,((break_time - start_time)*60)+1,30)
	tim_arr_after = np.arange(((break_time + break_time_duration)-start_time)*60, ((no_of_working_hrs - break_time_duration)*60)+1, 30)
	tim_arr = np.append(tim_arr_before,tim_arr_after)

	# Generate slots randomly
	start_times = np.random.choice(tim_arr,no_of_calls,replace=True)
	# Generate slot durations randomly
	duration = np.random.choice(np.array([30,60]),no_of_calls,replace=True)
	end_times = start_times + duration

	# Giving a unique id to all the customers
	customer_id = np.arange(1,(no_of_calls+1))

	# Creating the final combined table
	tim_loc_data = np.append(service_pos_arr,np.reshape(start_times,(-1,1)),axis=1)
	tim_loc_data = np.append(tim_loc_data,np.reshape(end_times,(-1,1)),axis=1)
	tim_loc_data = np.concatenate((tim_loc_data,customer_load),axis=1)
	tim_loc_data = np.insert(tim_loc_data,0,customer_id,axis=1)

	try:
		
		# Save the table	
		np.savetxt("time_loc_data.csv",tim_loc_data,delimiter=",")
	
	except PermissionError:
		print("The file is being used in another program or by another user. Please close the file and run this program again")
		print('CSV file could not be generated..........')
		print('The program has terminated')    
		sys.exit()

	print('CSV file generated succesfully...........')
	return tim_loc_data


if __name__ == '__main__':
	a = gen_data(26.9921975,75.7840554)