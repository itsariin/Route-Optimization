import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import folium
import osmnx as ox
import networkx as nx
import random

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
def f(a, b):
    try:
        d = nx.shortest_path_length(G, source=a, target=b, method='dijkstra', weight='travel_time')
    except:
        d = np.nan
    return d

distance_matrix = np.asarray([[f(a, b) for b in dtf["node"].tolist()] for a in dtf["node"].tolist()])
distance_matrix = pd.DataFrame(distance_matrix, columns=dtf["node"].values, index=dtf["node"].values)
distance_matrix = distance_matrix.round().astype(int)

# Replace NaN values in the distance matrix with a large number (e.g., 10000)
distance_matrix = distance_matrix.fillna(10000)

# Ant Colony Optimization (ACO) Algorithm with Iteration Tracking
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
        self.best_routes_per_iteration = []  # Store best route in each iteration

    def run(self):
        best_route = None
        best_distance = float('inf')
        all_time_best_route = None
        all_time_best_distance = float('inf')

        for iteration in range(self.num_iterations):
            all_routes = self.generate_all_routes()
            self.spread_pheromone(all_routes, self.num_ants, best_route=best_route)
            best_route, best_distance = min(all_routes, key=lambda x: x[1])

            if best_distance < all_time_best_distance:
                all_time_best_distance = best_distance
                all_time_best_route = best_route

            self.pheromone *= self.decay
            self.best_distances_per_iteration.append(best_distance)
            self.best_routes_per_iteration.append(best_route)  # Store best route per iteration

            print(f"Iteration {iteration + 1}/{self.num_iterations}, Best Distance: {best_distance}")

        return all_time_best_route, all_time_best_distance

    def generate_all_routes(self):
        all_routes = []
        for ant in range(self.num_ants):
            route = self.generate_route(0)  # Start at node 0
            distance = self.calculate_total_distance(route)
            all_routes.append((route, distance))
        return all_routes

    def generate_route(self, start_node):
        route = [start_node]
        visited = set(route)

        while len(visited) < len(self.distance_matrix):
            move_probs = self.calculate_move_probs(route[-1], visited)
            next_node = self.choose_next_node(move_probs)
            route.append(next_node)
            visited.add(next_node)

        route.append(start_node)
        return route

    def calculate_move_probs(self, current_node, visited):
        pheromone = np.copy(self.pheromone[current_node])
        pheromone[list(visited)] = 0  # Set pheromone of visited nodes to 0
        
        distances = np.copy(self.distance_matrix[current_node])
        inverse_distances = np.reciprocal(distances, where=distances != 0)  # Avoid division by zero
        
        move_probs = (pheromone ** self.alpha) * (inverse_distances ** self.beta)
        
        total_prob = move_probs.sum()
        
        # If total_prob is 0 (all distances are too large or pheromone is zero), select randomly
        if total_prob == 0:
            move_probs = np.ones_like(move_probs)
        
        return move_probs / move_probs.sum()

    def choose_next_node(self, move_probs):
        return np.random.choice(range(len(move_probs)), p=move_probs)

    def calculate_total_distance(self, route):
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += self.distance_matrix[route[i], route[i + 1]]
        return total_distance

    def spread_pheromone(self, all_routes, num_ants, best_route):
        for route, distance in all_routes:
            for i in range(len(route) - 1):
                self.pheromone[route[i], route[i + 1]] += 1 / distance

# Hyperparameters for ACO
num_ants = 10
num_iterations = 100
decay = 0.95
alpha = 1
beta = 2

# Run ACO on the distance matrix
ant_colony = AntColony(distance_matrix.values, num_ants, num_iterations, decay, alpha, beta)
optimized_route, optimized_distance = ant_colony.run()

# Animate the evolution of the route distances
fig, ax = plt.subplots()
ax.set_xlim(0, num_iterations)
ax.set_ylim(0, max(ant_colony.best_distances_per_iteration) * 1.1)

line, = ax.plot([], [], lw=2)
plt.title("ACO Route Optimization Progress")
plt.xlabel("Iteration")
plt.ylabel("Best Distance")

def init():
    line.set_data([], [])
    return line,

def update(frame):
    xdata = list(range(frame))
    ydata = ant_colony.best_distances_per_iteration[:frame]
    line.set_data(xdata, ydata)
    return line,

ani = FuncAnimation(fig, update, frames=num_iterations, init_func=init, blit=True)
plt.show()
