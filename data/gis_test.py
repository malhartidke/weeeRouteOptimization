import osmnx as ox
import geopandas as gpd
import numpy as np
import networkx as nx
import pandas as pd

fp = 'Sitapura.graphml' 
graph = ox.load_graphml(fp)

# train_file_location = 'time_loc_data.csv'
# df = pd.read_csv(train_file_location)
# tim_loc_data = np.array(df)
# service_pos_arr = tim_loc_data[:,1:3]

# path_file_location = 'best_sol.csv'
# dfp = pd.read_csv(path_file_location)                                 # Reading the CSV file
# path_data = np.array(dfp)

# main_lat = 26.7858865
# main_long = 75.8430487
# main_node = ox.get_nearest_node(graph,(main_lat,main_long),method='haversine')
# print(main_node)

# pathIds = path_data[4] - 1
# print(pathIds)
# if np.any(pathIds == -1):
# 	pathIds = pathIds[:-1]

# ids = ox.get_nearest_nodes(graph,service_pos_arr[pathIds,1],service_pos_arr[pathIds,0],method='balltree')
# ids = np.append(ids,main_node)
# ids = np.insert(ids,0,main_node)
# routes = []

# for idx in range(np.size(ids)-1):
# 	routes.append(nx.shortest_path(graph,source=ids[idx],target=ids[idx+1],weight='length'))

# fig, ax = ox.plot_graph_routes(graph, routes)
fig, ax = ox.plot_graph(graph, node_size=10, node_color='#66cc66')

'''
ids = np.asarray(ids)

fig, ax = ox.plot_graph(graph, node_size=30, node_color='#66cc66')
'''
#print(np.asarray(orig_node))
'''
#print(nodes_.head())
#
#print(orig_node)
'''
'''
convex_hull = edges.unary_union.convex_hull
area = convex_hull.area
stats = ox.basic_stats(graph, area=area)
extended_stats = ox.extended_stats(graph, ecc=True, cc=True)
for key, value in extended_stats.items():
    stats[key] = value
print(pd.Series(stats))
'''
