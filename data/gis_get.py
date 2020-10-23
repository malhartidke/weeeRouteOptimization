import osmnx as ox


# For VKI
# location_point = (26.9921975, 75.7840554)

# For Sitapura
location_point = (26.7858865, 75.8430487)

G = ox.graph_from_point(location_point, distance=4000, distance_type='bbox', network_type='drive')
ox.save_graphml(G, filename='Sitapura.graphml')