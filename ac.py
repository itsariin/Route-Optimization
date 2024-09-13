import numpy as np
from numpy import inf

# Given values for the problem
d = np.array([[0, 10, 12, 11, 14],
              [10, 0, 13, 15, 8],
              [12, 13, 0, 9, 14],
              [11, 15, 9, 0, 16],
              [14, 8, 14, 16, 0]])

iteration = 100
n_ants = 5
n_citys = 5

# Initialization
e = 0.5  # Evaporation rate
alpha = 1  # Pheromone factor
beta = 2  # Visibility factor

# Calculating the visibility of the next city visibility(i,j) = 1/d(i,j)
visibility = np.where(d != 0, 1 / d, 0)  # Avoid division by zero

# Initializing pheromones present at the paths to the cities
pheromone = 0.1 * np.ones((n_citys, n_citys))

# Initializing the route of the ants
rute = np.ones((n_ants, n_citys + 1), dtype=int)  # +1 for the return to the start

for ite in range(iteration):
    rute[:, 0] = 1  # Initial starting position of every ant (city 1)

    for i in range(n_ants):
        temp_visibility = np.copy(visibility)  # Create a copy of visibility
        visited = np.zeros(n_citys, dtype=bool)
        visited[0] = True  # Mark the starting city as visited

        for j in range(n_citys - 1):
            cur_loc = rute[i, j] - 1  # Current city (0-indexed)
            temp_visibility[:, cur_loc] = 0  # Set visibility of the current city to 0

            p_feature = pheromone[cur_loc, :] ** alpha
            v_feature = temp_visibility[cur_loc, :] ** beta

            combine_feature = p_feature * v_feature
            total = np.sum(combine_feature)
            if total == 0:
                # In case all features are zero, fallback to a uniform probability
                probs = np.ones(n_citys) / n_citys
            else:
                probs = combine_feature / total
            cum_prob = np.cumsum(probs)

            r = np.random.random()
            next_city = np.nonzero(cum_prob > r)[0][0] + 1
            rute[i, j + 1] = next_city

            visited[next_city - 1] = True

        # Ensure that all cities are visited and handle any issues
        if np.any(~visited):
            unvisited = np.nonzero(~visited)[0][0] + 1
        else:
            # If all cities are visited, choose the first city to return to start
            unvisited = rute[i, 0]
        rute[i, -2] = unvisited  # Add the last untraversed city to the route

    # Calculate route costs
    dist_cost = np.zeros(n_ants)
    for i in range(n_ants):
        route_cost = 0
        for j in range(n_citys):
            route_cost += d[rute[i, j] - 1, rute[i, (j + 1) % n_citys] - 1]
        dist_cost[i] = route_cost

    # Update pheromones
    best_ant = np.argmin(dist_cost)
    best_route = rute[best_ant]
    best_cost = dist_cost[best_ant]

    pheromone = (1 - e) * pheromone
    for i in range(n_ants):
        for j in range(n_citys):
            start_city = rute[i, j] - 1
            end_city = rute[i, (j + 1) % n_citys] - 1
            pheromone[start_city, end_city] += 1 / dist_cost[i]

print('Route of all the ants at the end:')
print(rute)
print()
print('Best path:', best_route)
print('Cost of the best path:', int(best_cost))
