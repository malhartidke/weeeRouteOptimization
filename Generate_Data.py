import osmnx as ox
import geopandas as gpd
import numpy as np
import networkx as nx
import itertools

def gen_data(lat,lon):

	print('\nGenerating CSV file..........')
	fp = 'VKI.graphml'
	graph = ox.load_graphml(fp)
	graph.remove_nodes_from(nx.isolates(graph))
	#fig, ax = ox.plot_graph(graph, node_size=30, node_color='#66cc66')
	nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
	
	main_node = ox.get_nearest_node(graph,(lat,lon),method='haversine')
	possibleNodes = np.asarray(nodes)[:,2]
	actualNodes = list(map(nx.has_path,itertools.repeat(graph,len(possibleNodes)),itertools.repeat(main_node,len(possibleNodes)),possibleNodes))
	actualIdx = list(itertools.compress(range(len(actualNodes)), actualNodes))
	finalNodes = nodes.iloc[actualIdx,:]

	# Number of calls recieved in a day are taken at random from a normal distribution with defined mean and standard deviation
	mu_c, sigma_c = 70,3 
	no_of_calls = int(np.random.normal(mu_c,sigma_c))
	randomChoiceArr = np.random.choice(finalNodes.shape[0],no_of_calls,replace=False)
	service_pos_arr = np.asarray(finalNodes.iloc[randomChoiceArr,:])[:,(0,1)]

	# The load present at the customer is taken at random from a normal distribution with defined mean and standard deviation
	mu_l, sigma_l = 6,1
	customer_load = np.random.normal(mu_l,sigma_l,no_of_calls).astype(int)
	customer_load = np.reshape(customer_load,(no_of_calls,1))

	# Giving a unique id to all the customers
	customer_id = np.arange(1,(no_of_calls+1))

	tim_loc_data = np.append(service_pos_arr,customer_load,axis=1)
	tim_loc_data = np.insert(tim_loc_data,0,customer_id,axis=1)
	#tim_loc_data = tim_loc_data[tim_loc_data[:,3].argsort()]

	try:
		np.savetxt("time_loc_data.csv",tim_loc_data,delimiter=",")             # File containing the data
	except PermissionError:
		print("The file is being used in another program or by another user. Please close the file and run this program again")
		print('CSV file could not be generated..........')
		print('The program has terminated')    
		sys.exit()
	print('CSV file generated succesfully...........')
	return tim_loc_data

if __name__ == '__main__':
	a = gen_data(26.9921975,75.7840554)