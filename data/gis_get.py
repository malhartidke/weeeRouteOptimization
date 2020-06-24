import osmnx as ox

location_point = (26.9921975, 75.7840554)
G = ox.graph_from_point(location_point, distance=2000, distance_type='bbox', network_type='drive')
ox.save_graphml(G, filename='VKI.graphml')