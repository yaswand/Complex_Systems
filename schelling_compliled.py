""" 
Schelling Model of Segregation
cater to groups
does not consider periodic condition to calculate clusters
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Patch

# from sklearn.cluster import DBSCAN
import networkx as nx

# Parameters
N = 50     # Grid side
E = 0.1     # Empty fraction
f1 = 0.5     # Group1 fraction
p = 0.7     # Fraction of satisfaction

time_steps = 100

def grid_setup(N, E, f1):
    total_cells = N * N
    empty_cells = int(total_cells * E)
    total_agents = total_cells - empty_cells
    group1_count = int(total_agents * f1)
    group2_count = total_agents - group1_count

    agents = np.array([1] * group1_count + [-1] * group2_count + [0] * empty_cells)
    np.random.shuffle(agents)
    grid = agents.reshape(N, N)
    print(grid)
    
    return grid

def get_adjacency_matrix(N):
    A = np.zeros((N*N, N*N))
    
    for row in range(N):
        for col in range(N):
            idx = row * N + col  # Convert 2D index to 1D
            
            neighbors = [
                ((row-1) % N, col), ((row+1) % N, col),  # Vertical neighbors
                (row, (col-1) % N), (row, (col+1) % N),  # Horizontal neighbors
                ((row-1) % N, (col-1) % N), ((row-1) % N, (col+1) % N),  # Diagonal neighbors
                ((row+1) % N, (col-1) % N), ((row+1) % N, (col+1) % N)
            ]
            
            for ni, nj in neighbors:
                A[idx][ni * N + nj] = 1
    
    return A

def compute_satisfaction(grid, N, A):
    q = grid.flatten()
    q1 = (q == 1).astype(int)
    q2 = (q == -1).astype(int)
    
    same_neighbors = (A @ q1) * q1 + (A @ q2) * q2
    total_neighbors = A @ (q != 0).astype(int)      # equivalent to q**2

    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.nan_to_num(same_neighbors / total_neighbors)  # Avoid division by zero
    
    return f.reshape(N, N)

def find_unsatisfied_agents(grid, f, p):
    return np.argwhere((grid != 0) & (f < p))

def find_empty_cells(grid):
    return np.argwhere(grid == 0)

def update_grid(grid, A, p):
    f = compute_satisfaction(grid, N, A)
    unsatisfied_agents = find_unsatisfied_agents(grid, f, p)
    empty_cells = find_empty_cells(grid)
    
    np.random.shuffle(unsatisfied_agents)
    
    empty_cells = list(map(tuple, empty_cells))  
    
    for x1, y1 in unsatisfied_agents:
        agent_value = grid[x1, y1]
        
        if np.random.rand() > 0.5:  # 50% chance to move to an empty cell
            if empty_cells:
                x2, y2 = empty_cells.pop(np.random.randint(len(empty_cells)))  
                grid[x2, y2], grid[x1, y1] = grid[x1, y1], 0
                continue  # Move to the next unsatisfied agent
        
        opposite_unsatisfied = unsatisfied_agents[grid[unsatisfied_agents[:, 0], unsatisfied_agents[:, 1]] == -agent_value]

        if len(opposite_unsatisfied) > 0:
            x2, y2 = opposite_unsatisfied[np.random.randint(len(opposite_unsatisfied))]
            grid[x2, y2], grid[x1, y1] = grid[x1, y1], grid[x2, y2]
            
    return grid

def run_simulation(N, E, f1, p, time_steps):
    grid = grid_setup(N, E, f1)
    A = get_adjacency_matrix(N)
    
    fig, ax = plt.subplots()
    im = ax.imshow(grid, cmap='coolwarm', interpolation='nearest')

    satisfaction_history = []
    
    def update(frame):
        nonlocal grid
        grid = update_grid(grid, A, p)
        im.set_array(grid)

        f = compute_satisfaction(grid, N, A)
        satisfied_count = np.sum((grid != 0) & (f >= p))
        total_agents = np.count_nonzero(grid)
        percent_satisfied = (satisfied_count / total_agents) * 100 if total_agents > 0 else 0
        satisfaction_history.append(percent_satisfied)
        
        print(f"Timestep: {frame + 1}, Satisfaction: {percent_satisfied:.2f}%")
        return [im]
    
    ax.set_title(f"Schelling Model of Segregation\nN={N}, E={E}, f1={f1}, p={p}")
    legend_patches = [Patch(color='white', label='Empty'),
                       Patch(color='blue', label='Group1'),
                       Patch(color='red', label='Group2')]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.005))
    
    ani = animation.FuncAnimation(fig, update, frames=time_steps, interval=100, blit = True , repeat=False)
    plt.show()

    # Plot satisfaction history
    plt.figure()
    plt.plot(range(1, time_steps + 1), satisfaction_history[3:])  # starting from 3 to skip the initial repeated time steps
    plt.xlabel("Timestep")
    plt.ylabel("Percentage of Satisfied Agents")
    plt.title("Satisfaction Over Time")
    plt.show()

    f = compute_satisfaction(grid, N, A)
    unsatisfied_agents = find_unsatisfied_agents(grid, f, p)

    satisfied_count = np.sum((grid != 0) & (f >= p))
    unsatisfied_count = len(unsatisfied_agents)

    total_agents = np.count_nonzero(grid)

    percent_satisfied = (satisfied_count / total_agents) * 100 if total_agents > 0 else 0
    percent_unsatisfied = (unsatisfied_count / total_agents) * 100 if total_agents > 0 else 0

    print(f"Final Percentage of satisfied agents: {percent_satisfied:.2f}%")
    print(f"Final Percentage of unsatisfied agents: {percent_unsatisfied:.2f}%")

    print(grid)
    
    print("done")

    return grid

grid = run_simulation(N, E, f1, p, time_steps)
'''
####################################################
# Identify clusters using DBSCAN with periodic boundaries
group1_cells = np.argwhere(grid == 1)
group2_cells = np.argwhere(grid == -1)

def periodic_distance(u, v, N):
    du = np.abs(u - v)
    du[du > N / 2] = N - du[du > N / 2]
    return np.sqrt(np.sum(du**2))

def custom_distance(u, v, N=N):
    return periodic_distance(u, v, N)

if len(group1_cells) > 0:
    dbscan_g1 = DBSCAN(eps=1.5, min_samples=3, metric=lambda u, v: custom_distance(u, v, N))  # Adjust eps and min_samples as needed
    labels_g1 = dbscan_g1.fit_predict(group1_cells)
    
    unique_labels_g1 = np.unique(labels_g1)
    print("Group 1 Clusters:", unique_labels_g1)
    
    plt.figure()
    scatter = plt.scatter(group1_cells[:, 1], group1_cells[:, 0], c=labels_g1, cmap='viridis')
    plt.title("Group 1 Clusters (Periodic Boundaries)")
    # Add legend
    legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.gca().add_artist(legend1)
    plt.show()
    
if len(group2_cells) > 0:
    dbscan_g2 = DBSCAN(eps=1.5, min_samples=3, metric=lambda u, v: custom_distance(u, v, N))  # Adjust eps and min_samples as needed
    labels_g2 = dbscan_g2.fit_predict(group2_cells)
    
    unique_labels_g2 = np.unique(labels_g2)
    print("Group 2 Clusters:", unique_labels_g2)
    
    plt.figure()
    scatter = plt.scatter(group2_cells[:, 1], group2_cells[:, 0], c=labels_g2, cmap='plasma')
    plt.title("Group 2 Clusters (Periodic Boundaries)")
    # Add legend
    legend2 = plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.gca().add_artist(legend2)
    plt.show()
########################################################

def calculate_segregation_index(labels, N, E):
    if labels is None or len(labels) == 0:
        return 0  # No segregation if no agents or clusters
    
    unique_labels = np.unique(labels)
    cluster_counts = {}
    for label in unique_labels:
        if label != -1:  # Exclude noise points
            cluster_counts[label] = np.sum(labels == label)
    
    total_agents = sum(cluster_counts.values())
    if total_agents == 0:
        return 0  # No segregation if no agents
    
    sum_nc_squared = sum(count**2 for count in cluster_counts.values())
    
    segregation_index = (2 * sum_nc_squared) / (N**2 * (1 - E ))**2
    return segregation_index

segregation_index_g1 = calculate_segregation_index(labels_g1, N, E)
print(f"Segregation Index (Group 1): {segregation_index_g1:.4f}")

segregation_index_g2 = calculate_segregation_index(labels_g2, N, E)
print(f"Segregation Index (Group 2): {segregation_index_g2:.4f}")

occupied_coords = np.argwhere(grid != 0)
segregation_index_g_total = calculate_segregation_index(occupied_coords, N, E)
print(f"Overall Segregation Index: {segregation_index_g_total:.4f}")

#######################################################################################
'''
# Modularity Calculation
A = get_adjacency_matrix(N)
G = nx.from_numpy_array(A)
q_flat = grid.flatten()
agent_nodes = np.where(q_flat != 0)[0]
subgraph = G.subgraph(agent_nodes)

group1_nodes = {agent_nodes[i] for i, val in enumerate(q_flat[agent_nodes]) if val == 1}
group2_nodes = {agent_nodes[i] for i, val in enumerate(q_flat[agent_nodes]) if val == -1}

if group1_nodes and group2_nodes:
    communities = [group1_nodes, group2_nodes]
    modularity = nx.community.modularity(subgraph, communities)
    print(f"Modularity: {modularity:.4f}")
else:
    print("Modularity: Not calculated, only one group found.")
'''
# Visualize the network
pos = {node: (node % N, node // N) for node in agent_nodes} # use grid coordinates for position
colors = [('blue' if q_flat[node] == 1 else 'red') for node in agent_nodes]

plt.figure()
nx.draw_networkx_nodes(subgraph, pos, node_color=colors, node_size=50)
nx.draw_networkx_edges(subgraph, pos, alpha=0.5)
plt.title("Schelling Model Network")
plt.show()

print("done") '''

###################################################################################

