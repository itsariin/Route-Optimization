import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import osmnx as ox
import networkx as nx
from tqdm import tqdm

# Load and Filter Data
city = "London"
dtf = pd.read_csv('data_stores.csv')
dtf = dtf[dtf["City"] == city][["City", "Street Address", "Latitude", "Longitude"]].reset_index(drop=True)
dtf = dtf.reset_index().rename(columns={"index": "id", "Latitude": "y", "Longitude": "x"})

# Prepare Data for Visualization
start = dtf[dtf["id"] == 0][["y", "x"]].values[0]
G = ox.graph_from_point(start, dist=10000, network_type="drive")
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)

start_node = ox.distance.nearest_nodes(G, start[1], start[0])
dtf["node"] = dtf[["y", "x"]].apply(lambda x: ox.distance.nearest_nodes(G, x.iloc[1], x.iloc[0]), axis=1)
dtf = dtf.drop_duplicates("node", keep='first')

# Distance Matrix Calculation
def calculate_distance(a, b):
    try:
        return nx.shortest_path_length(G, source=a, target=b, weight='travel_time')
    except:
        return np.inf

distance_matrix = np.array([[calculate_distance(a, b) for b in dtf["node"]] for a in tqdm(dtf["node"], desc="Calculating distances")])
distance_matrix = pd.DataFrame(distance_matrix, columns=dtf["node"], index=dtf["node"])
distance_matrix = distance_matrix.round().astype(float)

# Ant Colony Optimization (ACO) Algorithm
class AntColony:
    def __init__(self, distance_matrix, num_ants, num_iterations, decay, alpha=1, beta=2):
        self.distance_matrix = distance_matrix
        self.pheromone = np.ones(self.distance_matrix.shape) / len(distance_matrix)
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.best_distances_per_iteration = []
        self.best_routes_per_iteration = []

    def run(self):
        best_route = None
        best_distance = float('inf')
        all_time_best_route = None
        all_time_best_distance = float('inf')

        for iteration in tqdm(range(self.num_iterations), desc="ACO Progress"):
            all_routes = self.generate_all_routes()
            self.spread_pheromone(all_routes, self.num_ants, best_route=best_route)
            best_route, best_distance = min(all_routes, key=lambda x: x[1])

            if best_distance < all_time_best_distance:
                all_time_best_distance = best_distance
                all_time_best_route = best_route

            self.pheromone *= self.decay
            self.best_distances_per_iteration.append(best_distance)
            self.best_routes_per_iteration.append(best_route)

        return all_time_best_route, all_time_best_distance

    def generate_all_routes(self):
        all_routes = []
        for _ in range(self.num_ants):
            route = self.generate_route(0)
            distance = self.calculate_total_distance(route)
            all_routes.append((route, distance))
        return all_routes

    def generate_route(self, start_node):
        route = [start_node]
        visited = set(route)
        while len(visited) < len(self.distance_matrix):
            move_probs = self.calculate_move_probs(route[-1], visited)
            next_node = np.random.choice(range(len(move_probs)), p=move_probs)
            route.append(next_node)
            visited.add(next_node)
        route.append(start_node)
        return route

    def calculate_move_probs(self, current_node, visited):
        pheromone = self.pheromone[current_node].copy()
        distances = self.distance_matrix[current_node].copy()
        
        pheromone[list(visited)] = 0
        move_probs = np.zeros(len(pheromone))
        
        for i in range(len(pheromone)):
            if i not in visited and distances[i] != 0:
                move_probs[i] = (pheromone[i] ** self.alpha) * ((1 / distances[i]) ** self.beta)
        
        total_prob = move_probs.sum()
        return move_probs / total_prob if total_prob > 0 else np.ones_like(move_probs) / len(move_probs)

    def calculate_total_distance(self, route):
        return sum(self.distance_matrix[route[i], route[i+1]] for i in range(len(route) - 1))

    def spread_pheromone(self, all_routes, num_ants, best_route):
        for route, distance in all_routes:
            for i in range(len(route) - 1):
                self.pheromone[route[i], route[i+1]] += 1 / distance

# Visualization functions
def plot_distance_evolution(ant_colony):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(ant_colony.best_distances_per_iteration) + 1), ant_colony.best_distances_per_iteration)
    plt.title("ACO Route Optimization Progress")
    plt.xlabel("Iteration")
    plt.ylabel("Best Distance (seconds)")
    plt.grid(True)
    plt.show()

def plot_optimized_route(G, dtf, optimized_route):
    route_coordinates = dtf.iloc[optimized_route][['y', 'x']].values
    route_nodes = dtf.iloc[optimized_route]['node'].values

    m = folium.Map(location=[route_coordinates[0][0], route_coordinates[0][1]], zoom_start=12)

    for i, coord in enumerate(route_coordinates):
        folium.Marker(
            location=[coord[0], coord[1]],
            popup=f"Stop {i}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)

    for i in range(len(route_nodes) - 1):
        try:
            path = nx.shortest_path(G, route_nodes[i], route_nodes[i+1], weight='travel_time')
            path_coords = [[G.nodes[node]['y'], G.nodes[node]['x']] for node in path]
            folium.PolyLine(locations=path_coords, weight=2, color='blue').add_to(m)
        except nx.NetworkXNoPath:
            print(f"No path found between nodes {route_nodes[i]} and {route_nodes[i+1]}")

    return m

# Hyperparameters for ACO
num_ants = 20
num_iterations = 100
decay = 0.95
alpha = 1
beta = 2

# Run ACO on the distance matrix
print("Starting ACO optimization...")
ant_colony = AntColony(distance_matrix.values, num_ants, num_iterations, decay, alpha, beta)
optimized_route, optimized_distance = ant_colony.run()

# Visualize results
print("Generating visualizations...")
plot_distance_evolution(ant_colony)
optimized_route_map = plot_optimized_route(G, dtf, optimized_route)
optimized_route_map.save("optimized_route_map.html")

print(f"Optimized route: {optimized_route}")
print(f"Optimized distance: {optimized_distance:.2f} seconds")
print("The optimized route map has been saved as 'optimized_route_map.html'")

# Calculate and print the improvement
initial_route = list(range(len(dtf))) + [0]
initial_distance = ant_colony.calculate_total_distance(initial_route)
improvement = (initial_distance - optimized_distance) / initial_distance * 100
print(f"Improvement over initial route: {improvement:.2f}%")