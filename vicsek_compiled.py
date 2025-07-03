'''
ID 5090 - Assignment 1 - Vicsek Model
Yaswand - NA22B010

Que1: Also store data in csv file.(The lines of code used to write into a csv file is commented.)
Que2: Calculating no. of neighbours each agent has is commented
Que3: Compute Polarization order parameter
Ques4: Monte Carlo

Choose the code - 1,2,3,4 to run the simulation.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#import csv

def vicsek_basic():
    
    N = 400
    L = 100
    v0 = 0.5
    R_I = 3
    n = np.pi * 0.1  # Noise
    time_steps = 500

    current_frame = 0

    
    position = np.random.uniform(0, L, (N, 2))
    direction = np.random.uniform(0, 2 * np.pi, N)  

    polarization_data = []

    def update_position(position, direction, L, v0):
        position += v0 * np.column_stack((np.cos(direction), np.sin(direction)))  # dt = 1
        position %= L  # Periodic boundary
        return position 

    def find_distance_matrix(positions, L):
        delta = np.abs(positions[:, np.newaxis, :] - positions[np.newaxis, :, :])
        delta = np.minimum(delta, L - delta)
        return np.sqrt((delta ** 2).sum(axis=-1))

    def update_direction(position, direction, R_I, n, L):
        distance_matrix = find_distance_matrix(position, L)
        neighbors_mask = (distance_matrix < R_I) & (distance_matrix >= 0)

        sum_sin = np.where(neighbors_mask, np.sin(direction), 0).sum(axis=1)
        sum_cos = np.where(neighbors_mask, np.cos(direction), 0).sum(axis=1)
        
        avg_direction = np.arctan2(sum_sin, sum_cos)
        
        # Handle agents with no neighbors
        no_neighbors_mask = (neighbors_mask.sum(axis=1) == 0)
        avg_direction[no_neighbors_mask] = direction[no_neighbors_mask]

        avg_direction += np.random.uniform(-n / 2, n / 2, size=N)   # Adding noise

        # polarization
        polar_sum_sin = np.sum(np.sin(direction))
        polar_sum_cos = np.sum(np.cos(direction))
        polarization_parameter = np.sqrt(polar_sum_sin**2 + polar_sum_cos**2) / N
        polar_direction = np.arctan2(polar_sum_sin, polar_sum_cos)

        return avg_direction, polarization_parameter, polar_direction

    # Visualization 
    fig, ax = plt.subplots(figsize=(10, 10))
    scat = ax.scatter(position[:, 0], position[:, 1], c='blue', s=5)

    quiver = ax.quiver(
        position[:, 0], position[:, 1],
        np.cos(direction), np.sin(direction),
        angles='xy', scale_units='xy', scale=0.3, color='red'
    )

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal')


    ax.set_title("Vicsek Model", pad = 40)
    parameters_text = ax.text(
        0.5, 1.02,   # (x, y) position in figure coordinates
        f"N = {N} ,  L = {L} ,  v0 = {v0} ,  R_I = {R_I} ,  Noise_Strength = {round(n, 3)}", 
        fontsize=12, color='black', transform=ax.transAxes, ha='center'
    )
    '''
    # Open CSV file for writing
    csv_file = open("agent_positions.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Frame", "Agent_ID", "X", "Y", "Direction"])  # Header row
    '''

    def update(frame):  
        nonlocal position, direction, polarization_data  

        direction, polarization_parameter, polar_direction = update_direction(position, direction, R_I, n, L)
        position = update_position(position, direction, L, v0)

        polarization_data.append([polarization_parameter, polar_direction])

        print(f"Frame: {frame}, Polarization Magnitude: {polarization_parameter}, Polarization Direction: {polar_direction}")

        scat.set_offsets(position)
        quiver.set_offsets(position)
        quiver.set_UVC(np.cos(direction), np.sin(direction))
        return scat, quiver

    ani = FuncAnimation(fig, update, frames=time_steps, interval=50, blit=True, repeat=False)

    plt.show()

    #csv_file.close()

    # Compute average polarization order parameter using last 50 entries
    if len(polarization_data) >= 200:
        last_200 = np.array(polarization_data[-50:])
        avg_polarization_magnitude = np.mean(last_200[:, 0])
        avg_polarization_direction = np.mean(last_200[:, 1])
        print("\n=== Final Results ===")
        print(f"Average Polarization Magnitude (last 50 frames): {avg_polarization_magnitude}")
        print(f"Average Polarization Direction (last 50 frames): {avg_polarization_direction}")
    else:
        print("Not enough frames to compute last 50 averages.")

#"----------------------------------------------------------------------------------------------------------------------------------"

def vicsek_diagnos():

    N = 400
    L = 100
    v0 = 3
    R_I = 1
    n = 0.3  # Noise
    time_steps = 1000

    position = np.random.uniform(0, L, (N, 2))
    direction = np.random.uniform(0, 2 * np.pi, N)

    delta_theta_storage = []

    polarization_x_storage = []
    polarization_y_storage = []


    def update_position(position, direction, L, v0):
        position += v0 * np.column_stack((np.cos(direction), np.sin(direction)))  # dt = 1
        position %= L  # Periodic boundary
        return position


    def find_distance_matrix(positions, L):
        delta = np.abs(positions[:, np.newaxis, :] - positions[np.newaxis, :, :])
        delta = np.minimum(delta, L - delta)
        return np.sqrt((delta ** 2).sum(axis=-1))


    def update_direction(position, direction, R_I, n, L):
        distance_matrix = find_distance_matrix(position, L)
        neighbors_mask = (distance_matrix < R_I) & (distance_matrix >= 0)

        sum_sin = np.where(neighbors_mask, np.sin(direction), 0).sum(axis=1)
        sum_cos = np.where(neighbors_mask, np.cos(direction), 0).sum(axis=1)

        avg_direction = np.arctan2(sum_sin, sum_cos)

        # Handle agents with no neighbors
        no_neighbors_mask = (neighbors_mask.sum(axis=1) == 0)
        avg_direction[no_neighbors_mask] = direction[no_neighbors_mask]

        avg_direction += np.random.uniform(-n / 2, n / 2, size=N)  # Adding noise

        # polarization
        polar_sum_sin = np.sum(np.sin(direction))
        polar_sum_cos = np.sum(np.cos(direction))

        polarization_parameter = np.sqrt(polar_sum_sin ** 2 + polar_sum_cos ** 2) / N  # |p|
        #print(f"Polarization Parameter Magnitude: {polarization_parameter}")

        # Compute delta theta
        polar_direction = np.arctan2(polar_sum_sin, polar_sum_cos)  # Direction of polarization vector
        delta_theta = (polar_direction - avg_direction) % (2 * np.pi)
        # print( polar_direction, delta_theta)

        return avg_direction, delta_theta, polar_sum_cos/N, polar_sum_sin/N


    # Visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    scat = ax.scatter(position[:, 0], position[:, 1], c='blue', s=5)  # Particles
    quiver = ax.quiver(position[:, 0], position[:, 1], np.cos(direction), np.sin(direction),
                    angles='xy', scale_units='xy', scale=0.3, color='red')  # Arrows

    ax.set_xlim(0, L)
    ax.set_aspect('equal')
    ax.set_title("Vicsek Model")


    # Animation
    def update(frame):
        nonlocal position, direction
        direction, delta_theta, px, py = update_direction(position, direction, R_I, n, L)
        position = update_position(position, direction, L, v0)

        # Store delta theta values for histogram after simulation
        delta_theta_storage.extend(delta_theta)  # Accumulate values across frames
        polarization_x_storage.append(px)
        polarization_y_storage.append(py)

        print(f"Frame: {frame}, Polarization Magnitude: {np.mean(np.sqrt(np.array(px) ** 2 + np.array(py) ** 2))}, Polarization Direction: {np.arctan2(np.mean(py), np.mean(px))}")

        # Update visualization
        scat.set_offsets(position)
        quiver.set_offsets(position)
        quiver.set_UVC(np.cos(direction), np.sin(direction))
        return scat, quiver

    ani = FuncAnimation(fig, update, frames=time_steps, interval=50, blit=True, repeat=False)
    plt.show()  

    #printing average polarization order parameter for the realization
    last_50_px = polarization_x_storage[-50:]
    last_50_py = polarization_y_storage[-50:]

    avg_polarization_magnitude = np.mean(np.sqrt(np.array(last_50_px) ** 2 + np.array(last_50_py) ** 2))
    avg_polarization_direction = np.arctan2(np.mean(last_50_py), np.mean(last_50_px))

    print(f"Average Polarization Magnitude (last 50 frames): {avg_polarization_magnitude}")
    print(f"Average Polarization Direction (last 50 frames): {avg_polarization_direction}")


    # After simulation, plot histogram with all accumulated Δθ values and 2D histogram
    def plot_histograms(delta_theta_values, px_values, py_values):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))  

        # delta theta Histogram
        axes[0].hist(delta_theta_values, bins=90, edgecolor='black', alpha=0.7, label="Δθ Distribution")
        axes[0].set_xlabel("Δθ (Difference from Polarization Direction)")
        axes[0].set_ylabel("Count of Agents (Summed Over All Frames)")
        axes[0].set_title("Histogram of Δθ Values (Over Entire Simulation)")
        axes[0].grid(True, linestyle='--', alpha=0.6)
        #axes[0].set_xlim(-3.14, 3.14)
        

        # 2D Histogram of Px and Py
        hist = axes[1].hist2d(px_values, py_values, bins=50, cmap='Blues')
        axes[1].set_xlabel("Px (Normalized Polarization in x-direction)")
        axes[1].set_ylabel("Py (Normalized Polarization in y-direction)")
        axes[1].set_title("2D Histogram of Polarization Components")
        axes[1].grid(True, linestyle='--', alpha=0.6)
        fig.colorbar(hist[3], ax=axes[1], label="Counts") # Colorbar
        # axes[1].set_xlim(-1, 1)
        # axes[1].set_ylim(-1, 1)

        # Polarization magnitude histogram
        polarization_magnitude = np.sqrt(np.array(px_values)**2 + np.array(py_values)**2)
        axes[2].hist(polarization_magnitude, bins=50, edgecolor='black', alpha=0.7, label="|P| Distribution")
        axes[2].set_xlabel("|P| (Magnitude of Normalized Polarization)")
        axes[2].set_ylabel("Count of Frames") # Changed y-axis label
        axes[2].set_title("Histogram of Polarization Magnitude")
        axes[2].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()


    # Plot histograms after simulation ends
    plot_histograms(delta_theta_storage, polarization_x_storage, polarization_y_storage)

#"----------------------------------------------------------------------------------------------------------------------------------"

def vicsek_monte_carlo():

    # Parameters
    N = 400  
    L = 100 
    v0 = 1.5 
    R_I = 5  
    time_steps = 500 
    noise_degrees = np.arange(0, 181, 10)  # Noise strengths
    num_realizations = 10  # Number of independent runs for each noise level


    def update_position(position, direction, L, v0):
        position += v0 * np.column_stack((np.cos(direction), np.sin(direction)))  # dt = 1
        position %= L  # periodic boundary 
        return position


    def find_distance_matrix(positions, L):
        delta = np.abs(positions[:, np.newaxis, :] - positions[np.newaxis, :, :])
        delta = np.minimum(delta, L - delta)  #  periodic boundary 
        return np.sqrt((delta ** 2).sum(axis=-1))


    def update_direction(position, direction, R_I, noise_strength, L):
        distance_matrix = find_distance_matrix(position, L)
        neighbors_mask = (distance_matrix < R_I) & (distance_matrix >= 0)

        sum_sin = np.where(neighbors_mask, np.sin(direction), 0).sum(axis=1)
        sum_cos = np.where(neighbors_mask, np.cos(direction), 0).sum(axis=1)

        avg_direction = np.arctan2(sum_sin, sum_cos)

        # Handle agents with no neighbors
        no_neighbors_mask = (neighbors_mask.sum(axis=1) == 0)
        avg_direction[no_neighbors_mask] = direction[no_neighbors_mask]

        n = np.pi * noise_strength / 180  # Convert degrees to radians
        avg_direction += np.random.uniform(-n / 2, n / 2, size=N)

        # polarization
        polar_sum_sin = np.sum(np.sin(direction))
        polar_sum_cos = np.sum(np.cos(direction))
        polarization_parameter = np.sqrt(polar_sum_sin**2 + polar_sum_cos**2) / N

        return avg_direction, polarization_parameter

    def monte_carlo_simulation():
        polarization_results = []

        for noise in noise_degrees:
            print(f"Running simulations for noise: {noise}°")

            realization_polarizations = []  # Store polarizations for each realization

            for realization in range(num_realizations):
                print(f"  Realization {realization + 1}/{num_realizations}")

                # Initialize positions and directions for each realization
                position = np.random.uniform(0, L, (N, 2))
                direction = np.random.uniform(0, 2 * np.pi, N)
                polarization_over_time = []

                # Simulation loop (no animation for multiple realizations)
                for _ in range(time_steps):
                    direction, polarization_parameter = update_direction(position, direction, R_I, noise, L)
                    position = update_position(position, direction, L, v0)
                    polarization_over_time.append(polarization_parameter)

                # Compute long-time polarization for this realization
                expected_polarization = np.mean(polarization_over_time[-50:])
                realization_polarizations.append(expected_polarization)

                print("avg_polarization (last 50 frames)",expected_polarization)

            # Average polarization over all realizations for this noise level
            avg_polarization = np.mean(realization_polarizations)
            polarization_results.append(avg_polarization)
            print(f"Noise: {noise}°, Average Expected Polarization (over {num_realizations} realizations): {avg_polarization:.4f}")

        # Plot expected polarization vs. noise strength
        plt.figure(figsize=(8, 6))
        plt.plot(noise_degrees, polarization_results, marker='o', linestyle='-', color='b')

        # Add annotations for each point
        for x, y in zip(noise_degrees, polarization_results):
            plt.annotate(f"({x}, {y:.3f})", (x, y), textcoords="offset points", xytext=(0,10), ha='center')  # Adjust offset as needed

        plt.xlabel("Noise Strength (Degrees)")
        plt.ylabel("Expected Polarization")
        plt.title("Expected Polarization vs Noise Strength (Averaged over Realizations)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    # Run Monte Carlo Simulation
    monte_carlo_simulation()

#"----------------------------------------------------------------------------------------------------------------------------------"

def main():
    while True:  
        print("\nChoose code to run:")
        print(" 1: Ques 1,3")
        print(" 2: Ques 2 - diagnostic analysis")
        print(" 3: Ques 4 - monte carlo")
        print(" 4: Exit")

        choice = input("Enter your choice (1, 2, or 3): ")

        try:
            choice = int(choice)  # Convert the input to an integer

            if choice == 1:
                vicsek_basic()
            elif choice == 2:
                vicsek_diagnos()
            elif choice == 3:
                vicsek_monte_carlo()
            elif choice == 4:
                print("Exit")
                break  
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

        except ValueError:  
            print("Invalid input. Please enter a number.")


if __name__ == "__main__":
    main()

#"----------------------------------------------------------------------------------------------------------------------------------"

"""
Basic Vicsek Code - Runs animation for a single realization for 500 frames

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
N = 4
L = 10
v0 = 0.1
R_I = 3
n = np.pi * 0.2 # Noise
time_steps = 500

# Random positions and directions
position = np.random.uniform(0, L, (N, 2))
direction = np.random.uniform(0, 2 * np.pi, N)

def update_position(position, direction, L, v0):
    position += v0 * np.column_stack((np.cos(direction), np.sin(direction)))  # dt = 1
    position %= L  # Periodic boundary
    return position 

def find_distance_matrix(positions, L):
    delta = np.abs(positions[:, np.newaxis, :] - positions[np.newaxis, :, :])
    delta = np.minimum(delta, L - delta)
    return np.sqrt((delta ** 2).sum(axis=-1))

def update_direction(position, direction, R_I, n, L):
    distance_matrix = find_distance_matrix(position, L)
    neighbors_mask = (distance_matrix < R_I) & (distance_matrix >= 0)

    sum_sin = np.where(neighbors_mask, np.sin(direction), 0).sum(axis=1)
    sum_cos = np.where(neighbors_mask, np.cos(direction), 0).sum(axis=1)
    
    avg_direction = np.arctan2(sum_sin, sum_cos)
    
    # Handle agents with no neighbors
    no_neighbors_mask = (neighbors_mask.sum(axis=1) == 0)
    avg_direction[no_neighbors_mask] = direction[no_neighbors_mask]

    avg_direction += np.random.uniform(-n / 2, n / 2, size=N)   # Adding noise

    # polarization
    polar_sum_sin = np.sum(np.sin(direction))
    polar_sum_cos = np.sum(np.cos(direction))
    polarization_parameter = np.sqrt(polar_sum_sin**2 + polar_sum_cos**2) / N
    polar_direction = np.arctan2(polar_sum_sin, polar_sum_cos)

    return avg_direction, polarization_parameter, polar_direction

# Visualization 
fig, ax = plt.subplots(figsize=(10, 10))
scat = ax.scatter(position[:, 0], position[:, 1], c='blue', s=5)

quiver = ax.quiver(
    position[:, 0], position[:, 1],
    np.cos(direction), np.sin(direction),
    angles='xy', scale_units='xy', scale=1, color='red'
)

ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_aspect('equal')
# Display initialized parameters as text
ax.set_title("Vicsek Model", pad = 40)
parameters_text = ax.text(
    0.5, 1.02,   # (x, y) position in figure coordinates
    f"N = {N} ,  L = {L} ,  v0 = {v0} ,  R_I = {R_I} ,  Noise_Strength = {round(n, 3)}", 
    fontsize=12, color='black', transform=ax.transAxes, ha='center'
)
# Animation
def update(_):  # Underscore (_) since FuncAnimation passes a frame number we don't use
    global position, direction
    direction, polarization_parameter, polar_direction = update_direction(position, direction, R_I, n, L)
    position = update_position(position, direction, L, v0)

    print(f"Polarization Magnitude: {polarization_parameter}, Polarization Direction: {polar_direction}")

    # Update visualization
    scat.set_offsets(position)
    quiver.set_offsets(position)
    quiver.set_UVC(np.cos(direction), np.sin(direction))

    return scat, quiver

ani = FuncAnimation(fig, update, frames=time_steps, interval=50, blit=True, repeat=False)

plt.show()

"""