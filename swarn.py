import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from functools import lru_cache
import networkx as nx
import osmnx as ox

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
@lru_cache(maxsize=None)
def calculate_distance(a, b):
    try:
        return nx.shortest_path_length(G, source=a, target=b, weight='travel_time')
    except:
        return np.inf

distance_matrix = np.array([[calculate_distance(a, b) for b in dtf["node"]] for a in tqdm(dtf["node"], desc="Calculating distances")])
distance_matrix = pd.DataFrame(distance_matrix, columns=dtf["node"], index=dtf["node"])
distance_matrix = distance_matrix.round().astype(float)

# Particle Swarm Optimization (PSO) Algorithm
# Convert distance_matrix to NumPy array after it is created
distance_matrix = np.array(distance_matrix)

class Particle:
    def __init__(self, route, distance_matrix):
        self.route = route
        self.distance_matrix = distance_matrix  # This is now a NumPy array
        self.best_route = route[:]
        self.best_distance = self.calculate_total_distance()
        self.velocity = []

    def calculate_total_distance(self):
        # Integer-based indexing for NumPy array
        return sum(self.distance_matrix[self.route[i], self.route[i + 1]] for i in range(len(self.route) - 1))

    def update_personal_best(self):
        current_distance = self.calculate_total_distance()
        if current_distance < self.best_distance:
            self.best_route = self.route[:]
            self.best_distance = current_distance

    def apply_velocity(self):
        for swap in self.velocity:
            i, j = swap
            self.route[i], self.route[j] = self.route[j], self.route[i]

    def generate_velocity(self, global_best_route, c1=1.5, c2=1.5):
        velocity = []
        for i in range(len(self.route)):
            if self.route[i] != self.best_route[i]:
                velocity.append((i, self.route.index(self.best_route[i])))
            if self.route[i] != global_best_route[i]:
                velocity.append((i, self.route.index(global_best_route[i])))
        random.shuffle(velocity)
        self.velocity = velocity[:int(c1 + c2)]  # limit the velocity size

class PSO:
    def __init__(self, distance_matrix, num_particles, num_iterations, c1=1.5, c2=1.5):
        self.distance_matrix = distance_matrix
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.c1 = c1
        self.c2 = c2
        self.particles = []
        self.global_best_route = None
        self.global_best_distance = float('inf')

    def initialize_particles(self):
        initial_route = list(range(len(dtf["node"]))) + [0]
        for _ in range(self.num_particles):
            route = initial_route[:]
            random.shuffle(route[1:-1])  # Randomize all nodes except start and end
            particle = Particle(route, self.distance_matrix)
            self.particles.append(particle)
            if particle.best_distance < self.global_best_distance:
                self.global_best_route = particle.best_route[:]
                self.global_best_distance = particle.best_distance

    def run(self):
        self.initialize_particles()

        for iteration in range(self.num_iterations):
            for particle in self.particles:
                particle.generate_velocity(self.global_best_route, self.c1, self.c2)
                particle.apply_velocity()
                particle.update_personal_best()

                # Update global best if this particle has a better solution
                if particle.best_distance < self.global_best_distance:
                    self.global_best_route = particle.best_route[:]
                    self.global_best_distance = particle.best_distance

            print(f"Iteration {iteration + 1}: Global Best Distance = {self.global_best_distance}")

        return self.global_best_route, self.global_best_distance

# Parameters for PSO
num_particles = 50
num_iterations = 500
c1 = 2.0  # cognitive parameter (self-influence)
c2 = 2.0  # social parameter (influence of global best)

# Run PSO on the distance matrix
pso = PSO(distance_matrix, num_particles, num_iterations, c1, c2)
best_route, best_distance = pso.run()

print(f"Optimized route: {best_route}")
print(f"Optimized distance: {best_distance:.2f} seconds")

# Calculate and print the improvement
initial_route = list(range(len(dtf))) + [0]
initial_distance = Particle(initial_route, distance_matrix).calculate_total_distance()
improvement = (initial_distance - best_distance) / initial_distance * 100
print(f"Improvement over initial route: {improvement:.2f}%")
