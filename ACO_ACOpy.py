import folium
import networkx as nx
import osmnx as ox
import pandas as pd
import numpy as np
from tqdm import tqdm
from acopy import Colony, Solver

# Load and filter data
city = "London"
dtf = pd.read_csv('data_stores.csv')
dtf = dtf[dtf["City"] == city][["City", "Street Address", "Latitude", "Longitude"]].reset_index(drop=True)
dtf = dtf.reset_index().rename(columns={"index": "id", "Latitude": "y", "Longitude": "x"})

# Prepare data
start = dtf[dtf["id"] == 0][["y", "x"]].values[0]
G = ox.graph_from_point(start, dist=10000, network_type="drive")
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)

# Nearest nodes
start_node = ox.distance.nearest_nodes(G, start[1], start[0])
dtf["node"] = dtf[["y", "x"]].apply(lambda x: ox.distance.nearest_nodes(G, x.iloc[1], x.iloc[0]), axis=1)
dtf = dtf.drop_duplicates("node", keep='first')

# Distance matrix calculation
def calculate_distance(a, b):
    try:
        return nx.shortest_path_length(G, source=a, target=b, weight='travel_time')
    except:
        return np.inf

distance_matrix = np.array([[calculate_distance(a, b) for b in dtf["node"]] for a in tqdm(dtf["node"], desc="Calculating distances")])
distance_matrix = pd.DataFrame(distance_matrix, columns=dtf["node"], index=dtf["node"])
distance_matrix = distance_matrix.round().astype(float)

# Create a graph for ACOpy
graph = nx.Graph()
for i in range(len(dtf)):
    for j in range(i+1, len(dtf)):
        graph.add_edge(i, j, weight=distance_matrix.iloc[i, j])

# Set up the ACO colony and solver
colony = Colony(alpha=1, beta=3)  # alpha: pheromone importance, beta: distance importance
solver = Solver(rho=0.1, q=10)    # rho: pheromone evaporation rate, q: pheromone deposit amount

# Solve the problem using the graph directly
best_solution = solver.solve(graph, colony, limit=100)  # limit: number of iterations

# Extract the optimized route and cost
optimized_route = best_solution.nodes   # Use 'nodes' to access the path in ACOpy
optimized_distance = best_solution.cost  # 'cost' stores the total cost/distance

print(f"Optimized route: {optimized_route}")
print(f"Optimized distance: {optimized_distance:.2f} seconds")

# Improvement calculation
initial_route = list(range(len(dtf)))
initial_distance = sum(distance_matrix.iloc[initial_route[i], initial_route[i+1]] for i in range(len(initial_route)-1))
initial_distance += distance_matrix.iloc[initial_route[-1], initial_route[0]]  # Return to start
improvement = (initial_distance - optimized_distance) / initial_distance * 100
print(f"Improvement over initial route: {improvement:.2f}%")

# --- Visualization using Folium ---
def plot_optimized_route(G, dtf, optimized_route):
    # Extract the coordinates of the optimized route
    route_coordinates = dtf.iloc[optimized_route][['y', 'x']].values
    route_nodes = dtf.iloc[optimized_route]['node'].values

    # Initialize a Folium map centered on the first stop
    m = folium.Map(location=[route_coordinates[0][0], route_coordinates[0][1]], zoom_start=12)

    # Add markers for each stop
    for i, coord in enumerate(route_coordinates):
        folium.Marker(
            location=[coord[0], coord[1]],
            popup=f"Stop {i}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)

    # Plot the lines connecting the stops
    for i in range(len(route_nodes) - 1):
        try:
            path = nx.shortest_path(G, route_nodes[i], route_nodes[i+1], weight='travel_time')
            path_coords = [[G.nodes[node]['y'], G.nodes[node]['x']] for node in path]
            folium.PolyLine(locations=path_coords, weight=2, color='blue').add_to(m)
        except nx.NetworkXNoPath:
            print(f"No path found between nodes {route_nodes[i]} and {route_nodes[i+1]}")

    # Return the map object
    return m

# Generate and save the optimized route map
print("Generating optimized route map...")
optimized_route_map = plot_optimized_route(G, dtf, optimized_route)
optimized_route_map.save("optimized_route_map.html")
print("The optimized route map has been saved as 'optimized_route_map.html'")

