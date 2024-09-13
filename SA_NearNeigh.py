import pandas as pd
import numpy as np
import folium
import osmnx as ox
import networkx as nx
import random

# Loading and Filtering Data
city = "London"
dtf = pd.read_csv('data_stores.csv')
dtf = dtf[dtf["City"] == city][["City", "Street Address", "Latitude", "Longitude"]].reset_index(drop=True)
dtf = dtf.reset_index().rename(columns={"index": "id", "Latitude": "y", "Longitude": "x"})

print("total", len(dtf))
print(dtf.head(3))

# Preparing Data for Visualization
data = dtf.copy()
data["color"] = ''
data.loc[data['id'] == 0, 'color'] = 'red'
data.loc[data['id'] != 0, 'color'] = 'black'
start = data[data["id"] == 0][["y", "x"]].values[0]
print("starting point:", start)

# Building the Graph from Geographic Data
G = ox.graph_from_point(start, dist=10000, network_type="drive")
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)
fig, ax = ox.plot_graph(G, bgcolor="black", node_size=5, node_color="white", figsize=(16, 8))

# Mapping Locations to the Graph
start_node = ox.distance.nearest_nodes(G, start[1], start[0])
dtf["node"] = dtf[["y", "x"]].apply(lambda x: ox.distance.nearest_nodes(G, x[1], x[0]), axis=1)
dtf = dtf.drop_duplicates("node", keep='first')
print(dtf.head())

# Distance Matrix Calculation
def f(a, b):
    try:
        d = nx.shortest_path_length(G, source=a, target=b, method='dijkstra', weight='travel_time')
    except:
        d = np.nan
    return d if d != float('inf') else 99999  # Replace inf with a large number

distance_matrix = np.asarray([[f(a, b) for b in dtf["node"].tolist()] for a in dtf["node"].tolist()])
distance_matrix = pd.DataFrame(distance_matrix, columns=dtf["node"].values, index=dtf["node"].values)
distance_matrix = distance_matrix.round().astype(int)

# Create a mapping from node IDs to row/column indices in the distance matrix
node_to_index = {node: idx for idx, node in enumerate(distance_matrix.index)}

# Nearest Neighbor Heuristic
def nearest_neighbor_heuristic(dtf, start_node, distance_matrix):
    route = [start_node]
    total_distance = 0
    visited = set(route)

    current_node = start_node
    for _ in range(len(dtf) - 1):
        unvisited = {node: distance_matrix.at[current_node, node] for node in distance_matrix.columns if node not in visited}
        nearest_node = min(unvisited, key=unvisited.get)

        route.append(nearest_node)
        total_distance += unvisited[nearest_node]
        visited.add(nearest_node)

        current_node = nearest_node

    route.append(start_node)
    total_distance += distance_matrix.at[current_node, start_node]

    return route, total_distance

# Simulated Annealing for Route Optimization
def simulated_annealing(route, distance_matrix, initial_temp=1000, cooling_rate=0.995, max_steps=10000):
    def calculate_total_distance(route):
        total_distance = 0
        for i in range(len(route) - 1):
            idx_a = node_to_index[route[i]]
            idx_b = node_to_index[route[i + 1]]
            total_distance += distance_matrix.iloc[idx_a, idx_b]
        return total_distance

    def swap_two_nodes(route):
        a, b = random.sample(range(1, len(route) - 1), 2)
        route[a], route[b] = route[b], route[a]
        return route

    current_route = route.copy()
    current_distance = calculate_total_distance(current_route)
    best_route = current_route.copy()
    best_distance = current_distance

    temp = initial_temp
    for step in range(max_steps):
        new_route = swap_two_nodes(current_route.copy())
        new_distance = calculate_total_distance(new_route)

        if new_distance < best_distance:
            best_route = new_route.copy()
            best_distance = new_distance

        if new_distance < current_distance or random.random() < np.exp((current_distance - new_distance) / temp):
            current_route = new_route
            current_distance = new_distance

        temp *= cooling_rate
        if temp < 1e-3:
            break

    return best_route, best_distance

# Hybrid Heuristic + Simulated Annealing
initial_route, initial_distance = nearest_neighbor_heuristic(dtf, start_node, distance_matrix)
print(f"Initial Route: {initial_route}")
print(f"Initial Distance: {initial_distance/1000:.2f} km")

optimized_route, optimized_distance = simulated_annealing(initial_route, distance_matrix)
print(f"Optimized Route: {optimized_route}")
print(f"Optimized Distance: {optimized_distance/1000:.2f} km")

# Visualizing the Optimized Route
map = folium.Map(location=start, tiles="cartodbpositron", zoom_start=12)

# Adding markers for each location
for idx in range(len(dtf)):
    folium.Marker(
        location=[dtf.loc[dtf['node'] == dtf.iloc[idx]['node'], 'y'].values[0],
                  dtf.loc[dtf['node'] == dtf.iloc[idx]['node'], 'x'].values[0]],
        popup=f"{dtf.iloc[idx]['Street Address']}",
        icon=folium.Icon(color="red" if dtf.iloc[idx]['id'] == 0 else "black")
    ).add_to(map)

# Adding lines for the optimized route
for i in range(len(optimized_route) - 1):
    folium.PolyLine(
        locations=[dtf.loc[dtf['node'] == optimized_route[i], ['y', 'x']].values[0],
                   dtf.loc[dtf['node'] == optimized_route[i + 1], ['y', 'x']].values[0]],
        color="blue", weight=2.5, opacity=1
    ).add_to(map)

# Save the map to an HTML file
map.save("optimized_route_map.html")