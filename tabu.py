import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import lru_cache
import networkx as nx
import osmnx as ox

# Load and Filter Data
city = "London"
dtf = pd.read_csv('data_stores.csv')
dtf = dtf[dtf["City"] == city][["City", "Street Address", "Latitude", "Longitude"]].reset_index(drop=True)
dtf = dtf.reset_index().rename(columns={"index": "id", "Latitude": "y", "Longitude": "x"    })

# Prepare Data for Visualization
start = dtf[dtf["id"] == 0][["y", "x"]].values[0]
G = ox.graph_from_point(start, dist=10000, network_type="drive")
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)

start_node = ox.distance.nearest_nodes(G, start[1], start[0])
dtf["node"] = dtf[["y", "x"]].apply(lambda x: ox.distance.nearest_nodes(G, x.iloc[1], x.iloc[0]), axis=1)
dtf = dtf.drop_duplicates("node", keep='first')

# Distance Matrix Calculation
@lru_cache(maxsize=None)
def calculate_distance(a, b):
    try:
        return nx.shortest_path_length(G, source=a, target=b, weight='travel_time')
    except:
        return np.inf

distance_matrix = np.array([[calculate_distance(a, b) for b in dtf["node"]] for a in tqdm(dtf["node"], desc="Calculating distances")])
distance_matrix = pd.DataFrame(distance_matrix, columns=dtf["node"], index=dtf["node"])
distance_matrix = distance_matrix.round().astype(float)

# Tabu Search Algorithm
class TabuSearch:
    def __init__(self, distance_matrix, initial_solution, tabu_size=50, max_iterations=500, aspiration_criteria=0.1):
        self.distance_matrix = distance_matrix
        self.solution = initial_solution
        self.best_solution = initial_solution
        self.best_distance = self.calculate_total_distance(initial_solution)
        self.tabu_list = []
        self.tabu_size = tabu_size
        self.max_iterations = max_iterations
        self.aspiration_criteria = aspiration_criteria

    def calculate_total_distance(self, route):
        return sum(self.distance_matrix[route[i], route[i+1]] for i in range(len(route) - 1))

    def generate_neighbors(self, route):
        neighbors = []
        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route) - 1):
                neighbor = route[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]  # Swap two nodes
                neighbors.append(neighbor)
        return neighbors

    def is_tabu(self, route):
        return route in self.tabu_list

    def update_tabu_list(self, route):
        self.tabu_list.append(route)
        if len(self.tabu_list) > self.tabu_size:
            self.tabu_list.pop(0)

    def run(self):
        stagnation_count = 0
        for iteration in range(self.max_iterations):
            neighbors = self.generate_neighbors(self.solution)
            best_neighbor = None
            best_neighbor_distance = float('inf')

            # Evaluate neighbors
            for neighbor in neighbors:
                distance = self.calculate_total_distance(neighbor)
                if (distance < best_neighbor_distance and
                    (distance < self.best_distance or not self.is_tabu(neighbor))):
                    best_neighbor = neighbor
                    best_neighbor_distance = distance

            # Aspiration criteria: Allow tabu moves if they improve the best solution
            if best_neighbor_distance < self.best_distance * (1 - self.aspiration_criteria):
                self.best_solution = best_neighbor
                self.best_distance = best_neighbor_distance
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Update solution and tabu list
            self.solution = best_neighbor
            self.update_tabu_list(self.solution)

            # Stop if there is no improvement for several iterations
            if stagnation_count > 10:
                print("No improvement, terminating early.")
                break

            print(f"Iteration {iteration + 1}: Current Best Distance = {self.best_distance}")

        return self.best_solution, self.best_distance

# Initial Solution and Run Tabu Search
initial_solution = list(range(len(dtf["node"]))) + [0]
tabu_search = TabuSearch(distance_matrix.values, initial_solution)
best_route, best_distance = tabu_search.run()

print(f"Optimized route: {best_route}")
print(f"Optimized distance: {best_distance:.2f} seconds")

# Calculate and print the improvement
initial_distance = tabu_search.calculate_total_distance(initial_solution)
improvement = (initial_distance - best_distance) / initial_distance * 100
print(f"Improvement over initial route: {improvement:.2f}%")
