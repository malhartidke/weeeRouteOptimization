#import sys
import numpy as np
import osmnx as ox
import networkx as nx
#import pandas as pd
#from math import sqrt

def dist_mat(tim_loc_data,main_lat,main_long):
	print('\nCreating Distance Matrix......')
	min_dist = 0
	service_pos_arr = tim_loc_data[:,1:3]
	fp = 'VKI.graphml' 
	graph = ox.load_graphml(fp)
	main_node = ox.get_nearest_node(graph,(main_lat,main_long),method='haversine')
	ids = ox.get_nearest_nodes(graph,service_pos_arr[:,1],service_pos_arr[:,0],method='balltree')
	ids = np.asarray(ids)
	matr = np.zeros((np.size(tim_loc_data,0)+1,np.size(tim_loc_data,0)+1))
	for x in range(np.size(matr,0)):
		for y in range(x+1):
			if y == x:
				matr[x][y] = 0				
				continue
			else:
				if y == 0:
					try:
						matr[x][y] = nx.shortest_path_length(graph,source=main_node,target=ids[x-1],weight='length')#cost_calc(main_lat,main_long,service_pos_arr[x-1,0],service_pos_arr[x-1,1])
					except nx.exception.NetworkXNoPath:
						avgValue = 3500
						print('No Direct Path was found. Taking distance as average value of 3500 units')
						matr[x][y] = 3500					
				else:
					try:
						matr[x][y] = nx.shortest_path_length(graph,source=ids[x-1],target=ids[y-1],weight='length')#cost_calc(service_pos_arr[x-1,0],service_pos_arr[x-1,1],service_pos_arr[y-1,0],service_pos_arr[y-1,1])
					except nx.exception.NetworkXNoPath:
						avgValue = 3500
						print('No Direct Path was found. Taking distance as average value of 3500 units')
						matr[x][y] = 3500	
	ul = np.triu_indices(np.size(matr,0))
	matr[ul] = matr.T[ul]
	np.savetxt('Dist_Matrix.csv',matr,delimiter=",")
	print('Distance Matrix Created and Saved......')
	return matr

def form_edge_table(dist_matrix):
	n = np.size(dist_matrix,0)
	edge_table = np.zeros((dist_matrix.size,3))

	edge_table[:,2] = dist_matrix.flatten()

	col_2 = np.arange(0,n)
	sec_col = np.repeat(np.reshape(col_2,(1,col_2.size)),n,axis=0)
	edge_table[:,1] = sec_col.flatten()
	
	for mul in range(n):
		edge_table[(mul*n):((mul+1)*n),0] = np.ones(n)*mul

	edge_table = np.delete(edge_table, np.argwhere(edge_table[:,2] == 0).flatten(),axis=0)
	edge_table = np.delete(edge_table, np.argwhere(edge_table[:,1] == 0).flatten(),axis=0)
	return edge_table

def min_dist(matr):
	min_dist = 0
	for idx in range(np.size(matr,1)-1):
		col = matr[:,idx]
		min_dist = min_dist + col[np.where(col==np.min(col[np.nonzero(col)]))[0][0]]
	return min_dist

def distance_calc(customer_info,capacity,dist_mat):
	customer_load = customer_info[:,1]
	customer_id = customer_info[:,0]
	reached_home = 1
	total_dist = 0 
	total_load = 0

	for idx, c_id in enumerate(customer_id):
		if reached_home == 1:
			total_dist = total_dist + dist_mat[int(customer_id[idx])][0]
		else:
			total_dist = total_dist + dist_mat[int(customer_id[idx-1])][int(customer_id[idx])]
		total_load = total_load + customer_load[idx]
		if idx == (customer_id.size-1):
			total_dist = total_dist + dist_mat[int(customer_id[idx])][0]			
			break
		else:
			if (total_load+customer_load[idx+1]) <= capacity:				
				reached_home = 0
				continue
			else:
				total_dist = total_dist + dist_mat[int(customer_id[idx])][0]
				reached_home = 1
				total_load = 0
	return total_dist
'''
def min_distance_calc(tim_loc_data,main_lat,main_long,speed):
	service_pos_arr = tim_loc_data[:,1:3]
	customer_id = tim_loc_data[:,0]

	QgsApplication.setPrefixPath('/usr', True)
	qgs = QgsApplication([], False)
	qgs.initQgis()

	total_dist = 0

	for c_id in range(customer_id.size):
		total_dist = total_dist + cost_calc(main_lat,main_long,service_pos_arr[c_id,0],service_pos_arr[c_id,1],speed)

	return total_dist
'''
'''
if __name__ == '__main__':
	train_file_location = 'time_loc_data.csv'
	df = pd.read_csv(train_file_location)
	tim_loc_data = np.array(df)
	main_long = 75.7840554
	main_lat = 26.9921975
	a = dist_mat(tim_loc_data,main_lat,main_long)
'''