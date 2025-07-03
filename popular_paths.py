import numpy as np
import matplotlib.pyplot as plt
import random
import math
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from matplotlib.patches import Polygon
from collections import deque

# Constants
WIDTH, HEIGHT = 200, 200
BG_COLOR = (0.2, 0.7, 0.2)  # Green background
PATH_COLOR = (0.6, 0.6, 0.6)  # Gray for established paths
TURTLE_COLOR = (0.0, 0.0, 1.0)  # Blue for turtles
DESTINATION_COLOR = (1.0, 0.0, 0.0)  # Red for destinations
HOUSE_COLOR = (1.0, 0.5, 0.0)  # Orange for houses
CURRENT_TARGET_COLOR = (0.8, 0.0, 0.0)  # Bright red for current target house

# Model parameters (can be adjusted)
POPULARITY_DECAY_RATE = 4
POPULARITY_PER_STEP = 20
MINIMUM_ROUTE_POPULARITY = 80
WALKER_COUNT = 100
WALKER_VISION_DIST = 10
SHOW_POPULARITY = True
TURTLE_SPEED = 2
PATH_INFLUENCE = 0.6  # How much popular paths influence movement (0-1)
MIN_MOVE_DISTANCE = 0.5  # Minimum distance a turtle must move in each step

# Grid cell size
CELL_SIZE = 0.5

# Triangle parameters
TRIANGLE_SCALE = 1  # Scale factor for triangle size

# Global variables
tick = 0
paused = False
grid = []
turtles = []
turtle_x = []
turtle_y = []
turtle_dx = []  # Direction vectors for turtles
turtle_dy = []
dest_x = []
dest_y = []
grid_width = 0
grid_height = 0
houses = []  # List to store house locations
adding_houses = False  # Flag to indicate if we're in house-adding mode
triangle_artists = []  # Store triangle artists for each turtle
new_house_added = False # Flag to indicate if a new house was added
new_house_x = -1 # to store the new house x
new_house_y = -1 # to store the new house y

house_addition_time_steps = []

def init_patch(x, y):
    """Initialize a patch with default values"""
    return {
        'x': x,
        'y': y,
        'popularity': 0,
        'is_route': False,
        'last_visited': 0
    }

def init_turtle(x, y):
    """Initialize a turtle with house visiting as primary objective"""
    angle = random.uniform(0, 2 * math.pi)
    return {
        'x': x,
        'y': y,
        'speed': TURTLE_SPEED,
        'dx': math.cos(angle),  # Initial random direction
        'dy': math.sin(angle),
        'house_queue': deque(),  # Queue of houses to visit
        'current_house_index': -1,  # Index of current house being visited
        'all_houses_visited': False,  # Flag to track if all houses have been visited
        'houses_visited': set(),  # Set to track which houses have been visited
        'prev_position': (x, y),  # Track previous position to ensure movement
        'is_moving_to_house': False, # Flag to indicate if turtle is moving to a house
        'dest_x': None,  # Initialize destination x
        'dest_y': None   # Initialize destination y
    }

def assign_house_queue(turtle):
    """Assign a queue of all houses for the turtle to visit in a random order,
    ensuring the most recently visited house is not at the start."""
    global houses

    if len(houses) > 0:
        house_indices = list(range(len(houses)))
        random.shuffle(house_indices)

        # Get the index of the most recently visited house
        last_visited = -1
        if turtle['houses_visited']:
            # Assuming the last element added to the set was the most recently visited
            last_visited = list(turtle['houses_visited'])[-1]

        # If the shuffled list starts with the last visited house, rotate it
        if house_indices and house_indices[0] == last_visited and len(house_indices) > 1:
            # Rotate the list by one position
            first = house_indices.pop(0)
            house_indices.append(first)

        turtle['house_queue'] = deque(house_indices)
        turtle['all_houses_visited'] = False
    else:
        turtle['house_queue'] = deque()
        turtle['all_houses_visited'] = True

def set_next_house_destination(turtle):
    """Set the next house in the queue as the turtle's destination, cycling indefinitely.
    If the queue is empty, it will be repopulated with all houses.
    The most recently visited house is added to the end of the queue."""
    global houses

    if len(houses) > 0:
        if not turtle['house_queue']:
            # If the queue is empty, repopulate it with all house indices in a random order
            house_indices = list(range(len(houses)))
            random.shuffle(house_indices)
            turtle['house_queue'] = deque(house_indices)

        if turtle['house_queue']:
            house_index = turtle['house_queue'].popleft()
            turtle['current_house_index'] = house_index
            house = houses[house_index]
            turtle['dest_x'] = house['x']
            turtle['dest_y'] = house['y']
            turtle['is_moving_to_house'] = True

            # Add the *previously* current house (which is now being visited)
            # to the end of the queue
            if turtle['current_house_index'] != -1:
                turtle['house_queue'].append(turtle['current_house_index'])
        else:
            turtle['dest_x'] = None
            turtle['dest_y'] = None
            turtle['current_house_index'] = -1
            turtle['is_moving_to_house'] = False
    else:
        turtle['house_queue'] = deque()
        turtle['dest_x'] = None
        turtle['dest_y'] = None
        turtle['current_house_index'] = -1
        turtle['is_moving_to_house'] = False

def distance_between(x1, y1, x2, y2):
    """Calculate shortest distance with periodic boundaries"""
    dx = min(abs(x1 - x2), grid_width - abs(x1 - x2))
    dy = min(abs(y1 - y2), grid_height - abs(y1 - y2))
    return math.sqrt(dx**2 + dy**2)

def distance_to_destination(turtle):
    """Calculate distance from turtle to its destination"""
    if turtle['dest_x'] is not None and turtle['dest_y'] is not None:
        return distance_between(turtle['x'], turtle['y'], turtle['dest_x'], turtle['dest_y'])
    return float('inf') # If no destination, consider it far away

def ensure_movement(dx, dy, min_distance=MIN_MOVE_DISTANCE):
    """Ensure the movement vector produces meaningful movement"""
    # Calculate the magnitude of the movement vector
    magnitude = math.sqrt(dx*dx + dy*dy)

    # If magnitude is too small, boost it to the minimum distance
    if magnitude < min_distance:
        # Normalize the vector then scale to minimum distance
        if magnitude > 0:  # Avoid division by zero
            dx = (dx / magnitude) * min_distance
            dy = (dy / magnitude) * min_distance
        else:
            # If vector is (0,0), create a random direction vector
            angle = random.uniform(0, 2 * math.pi)
            dx = math.cos(angle) * min_distance
            dy = math.sin(angle) * min_distance

    return dx, dy

def update_grid_popularity(turtle, current_tick):
    """Update the popularity of the grid cell the turtle is currently on."""
    x, y = int(turtle['x']), int(turtle['y'])
    patch = grid[y][x]
    patch['popularity'] += POPULARITY_PER_STEP
    patch['last_visited'] = current_tick
    if patch['popularity'] >= MINIMUM_ROUTE_POPULARITY and not patch['is_route']:
        patch['is_route'] = True

def move_turtle(turtle, current_tick):
    """Move a turtle, either randomly or towards a house based on the number of houses."""
    global grid, houses

    prev_x, prev_y = turtle['x'], turtle['y']
    turtle['prev_position'] = (prev_x, prev_y)

    if len(houses) < 2:
        # If less than two houses, move randomly
        if random.random() < 0.05:
            angle = random.uniform(0, 2 * math.pi)
            turtle['dx'] = math.cos(angle)
            turtle['dy'] = math.sin(angle)
        turtle['dx'], turtle['dy'] = ensure_movement(turtle['dx'], turtle['dy'])
        turtle['x'] = (turtle['x'] + turtle['dx'] * turtle['speed']) % grid_width
        turtle['y'] = (turtle['y'] + turtle['dy'] * turtle['speed']) % grid_height
        update_grid_popularity(turtle, current_tick)
        turtle['is_moving_to_house'] = False # Ensure the flag is off
        turtle['current_house_index'] = -1 # Reset current target
        return

    # If two or more houses, navigate between them
    if not turtle['is_moving_to_house']:
        set_next_house_destination(turtle)

    if turtle['is_moving_to_house'] and turtle['current_house_index'] != -1:
        house = houses[turtle['current_house_index']]
        if distance_between(turtle['x'], turtle['y'], house['x'], house['y']) < 1:
            turtle['houses_visited'].add(turtle['current_house_index'])
            set_next_house_destination(turtle)

    if turtle['is_moving_to_house'] and turtle['current_house_index'] != -1:
        grid_x = int(turtle['x'])
        grid_y = int(turtle['y'])

        # Calculate direct vector to destination (house)
        dx_direct = house['x'] - turtle['x']
        dy_direct = house['y'] - turtle['y']

        # Handle periodic boundaries
        if abs(dx_direct) > grid_width / 2:
            dx_direct = -math.copysign(grid_width - abs(dx_direct), dx_direct)
        if abs(dy_direct) > grid_height / 2:
            dy_direct = -math.copysign(grid_height - abs(dy_direct), dy_direct)

        # Normalize direct vector
        direct_length = max(0.01, math.sqrt(dx_direct * dx_direct + dy_direct * dy_direct))
        dx_direct /= direct_length
        dy_direct /= direct_length

        # Initialize dx and dy with default values
        dx = dx_direct
        dy = dy_direct

        # Look for popular routes only in the forward direction
        most_popular_route = None
        most_popularity = -1

        # The direction angle based on the direct vector to the house.
        target_angle = math.atan2(dy_direct, dx_direct)

        # Search within a cone-shaped region in front of the turtle
        angle_offset_deg = -90  # Cone angle: +/- 45 degrees
        while angle_offset_deg <= 90:
            angle = target_angle + math.radians(angle_offset_deg)
            dist = 1
            while dist <= WALKER_VISION_DIST:
                check_x = int(grid_x + dist * math.cos(angle))
                check_y = int(grid_y + dist * math.sin(angle))
                check_x = check_x % grid_width
                check_y = check_y % grid_height

                patch = grid[check_y][check_x]

                if patch['is_route']:
                    if patch['popularity'] > most_popularity:
                        most_popular_route = (check_x, check_y)
                        most_popularity = patch['popularity']
                dist += 1
            angle_offset_deg += 10

        # Determine final movement direction
        if most_popular_route is not None:
            # Calculate vector to popular route
            pop_x, pop_y = most_popular_route
            dx_pop = pop_x - turtle['x']
            dy_pop = pop_y - turtle['y']

            # Handle periodic boundaries
            if abs(dx_pop) > grid_width / 2:
                dx_pop = -math.copysign(grid_width - abs(dx_pop), dx_pop)
            if abs(dy_pop) > grid_height / 2:
                dy_pop = -math.copysign(grid_height - abs(dy_pop), dy_pop)

            # Normalize popular route vector
            pop_length = max(0.01, math.sqrt(dx_pop * dx_pop + dy_pop * dy_pop))
            dx_pop /= pop_length
            dy_pop /= pop_length

            # Blend direct path and popular path, but with more influence from popular path
            dx = (1 - PATH_INFLUENCE * 0.5) * dx_direct + (PATH_INFLUENCE * 0.5) * dx_pop
            dy = (1 - PATH_INFLUENCE * 0.5) * dy_direct + (PATH_INFLUENCE * 0.5) * dy_pop

            # Normalize blended vector
            final_length = max(0.01, math.sqrt(dx * dx + dy * dy))
            dx /= final_length
            dy /= final_length

        # Ensure we're moving by a minimum distance
        dx, dy = ensure_movement(dx, dy)

        # Update turtle direction and position
        turtle['dx'] = dx
        turtle['dy'] = dy

        turtle['x'] += dx * turtle['speed']
        turtle['y'] += dy * turtle['speed']

        # Apply periodic boundaries
        turtle['x'] = turtle['x'] % grid_width
        turtle['y'] = turtle['y'] % grid_height

        # Update grid patch popularity
        new_grid_x = int(turtle['x'])
        new_grid_y = int(turtle['y'])

        patch = grid[new_grid_y][new_grid_x]
        patch['popularity'] += POPULARITY_PER_STEP
        patch['last_visited'] = current_tick

        if patch['popularity'] >= MINIMUM_ROUTE_POPULARITY and not patch['is_route']:
            patch['is_route'] = True


def reset_simulation(event=None):
    global tick, grid, turtles, turtle_x, turtle_y, turtle_dx, turtle_dy, dest_x, dest_y, triangle_artists, order_parameters_history, calculation_started, history_index,house_addition_time_steps

    tick = 0
    calculation_started = False
    order_parameters_history = None
    history_index = 0
    global house_addition_time_steps
    house_addition_time_steps = [] 

    # Initialize grid of patches
    grid = [[init_patch(x, y) for x in range(grid_width)] for y in range(grid_height)]

    # Create turtles
    turtles = []
    turtle_x = []
    turtle_y = []
    turtle_dx = []
    turtle_dy = []
    dest_x = []
    dest_y = []

    for _ in range(WALKER_COUNT):
        x = random.randint(0, grid_width - 1)
        y = random.randint(0, grid_height - 1)
        turtle = init_turtle(x, y)
        turtles.append(turtle)
        turtle_x.append(x)
        turtle_y.append(y)
        turtle_dx.append(turtle['dx']) # Initial random direction
        turtle_dy.append(turtle['dy']) # Initial random direction
        dest_x.append(turtle['dest_x'])
        dest_y.append(turtle['dest_y'])

    # Clear existing triangle artists
    for artist in triangle_artists:
        if artist in plt.gca().patches:
            artist.remove()
    triangle_artists = []

def toggle_pause(event=None):
    """Toggle pause state"""
    global paused
    paused = not paused

def toggle_house_mode(event=None):
    """Toggle house placement mode"""
    global adding_houses
    adding_houses = not adding_houses
    if adding_houses:
        print("House placement mode activated. Click on the grid to add/remove houses.")
    else:
        print("House placement mode deactivated.")

def find_house_at_position(x, y):
    """Find a house at the given position and return its index, or -1 if not found"""
    for i, house in enumerate(houses):
        if house['x'] == x and house['y'] == y:
            return i
    return -1

def on_click(event):
    global houses, new_house_added, new_house_x, new_house_y, turtles, calculation_started, order_parameters_history, history_index

    if event.inaxes:
        x, y = int(event.xdata), int(event.ydata)

        # Check if there's already a house at this location
        house_index = find_house_at_position(x, y)

        if house_index >= 0:
            # Remove existing house
            removed_house = houses.pop(house_index)
            print(f"House removed from ({removed_house['x']}, {removed_house['y']})")
            for turtle in turtles:
                # Update turtle's house queue and visited set
                new_queue = deque()
                for idx in turtle['house_queue']:
                    if idx == house_index:
                        pass
                    elif idx > house_index:
                        new_queue.append(idx - 1)
                    else:
                        new_queue.append(idx)
                turtle['house_queue'] = new_queue
                new_visited = set()
                for idx in turtle['houses_visited']:
                    if idx == house_index:
                        pass
                    elif idx > house_index:
                        new_visited.add(idx - 1)
                    else:
                        new_visited.add(idx)
                turtle['houses_visited'] = new_visited
                if turtle['current_house_index'] == house_index:
                    turtle['current_house_index'] = -1
                    set_next_house_destination(turtle)
                elif turtle['current_house_index'] > house_index:
                    turtle['current_house_index'] -= 1
        else:
           # Add new house
            houses.append({'x': x, 'y': y})
            new_house_index = len(houses) - 1
            print(f"House added at ({x}, {y}) with index {new_house_index}")
            
            # Record the time step when the house was added
            house_addition_time_steps.append(tick)
            print(f"House added at time step: {tick}")
            
            new_house_added = True
            new_house_x = x
            new_house_y = y

            # Point all agents to the new house and shuffle their queue
            for turtle in turtles:
                house_list = list(turtle['house_queue'])
                house_list.append(new_house_index)
                random.shuffle(house_list)
                turtle['house_queue'] = deque(house_list)
                turtle['houses_visited'] = set()
                set_next_house_destination(turtle)
                turtle['is_moving_to_house'] = True
            if len(houses) >= 2 and not calculation_started:
                calculation_started = True
                order_parameters_history = np.zeros((5000, 3)) # Re-initialize if starting now
                history_index = 0

def adjust_path_influence(event=None, increase=True):
    """Adjust the influence of popular paths on movement"""
    global PATH_INFLUENCE

    if increase:
        PATH_INFLUENCE = min(1.0, PATH_INFLUENCE + 0.1)
    else:
        PATH_INFLUENCE = max(0.0, PATH_INFLUENCE - 0.1)

    print(f"Path influence set to: {PATH_INFLUENCE:.1f}")

def update_turtle_destinations():
    """Update all turtle destinations based on current houses"""
    for turtle in turtles:
        # Reset the visited houses when we update destinations
        turtle['houses_visited'] = set()
        assign_house_queue(turtle)
        set_next_house_destination(turtle)
        turtle['is_moving_to_house'] = True # Ensure they start moving to houses

def clear_houses(event=None):
    global houses, turtles, order_parameters_history, calculation_started, history_index, house_addition_time_steps
    houses = []
    calculation_started = False
    order_parameters_history = None
    history_index = 0
    house_addition_time_steps = [] 
    print("All houses cleared. Turtles will now move randomly.")
    for turtle in turtles:
        turtle['house_queue'] = deque()
        turtle['current_house_index'] = -1
        turtle['is_moving_to_house'] = False
        turtle['dest_x'] = None
        turtle['dest_y'] = None

def on_key(event):
    """Handle keyboard shortcuts"""
    global WALKER_COUNT, PATH_INFLUENCE, MIN_MOVE_DISTANCE

    if event.key == ' ':
        toggle_pause()
    elif event.key == 'r':
        reset_simulation()
    elif event.key == 'up':
        WALKER_COUNT = min(1000, WALKER_COUNT + 10)
        reset_simulation()
    elif event.key == 'down':
        WALKER_COUNT = max(5, WALKER_COUNT - 10)
        reset_simulation()
    elif event.key == 'c':
        clear_houses()
    elif event.key == '+':
        adjust_path_influence(increase=True)
    elif event.key == '-':
        adjust_path_influence(increase=False)
    elif event.key == '>':
        MIN_MOVE_DISTANCE = min(2.0, MIN_MOVE_DISTANCE + 0.1)
        print(f"Minimum move distance: {MIN_MOVE_DISTANCE:.1f}")
    elif event.key == '<':
        MIN_MOVE_DISTANCE = max(0.1, MIN_MOVE_DISTANCE - 0.1)
        print(f"Minimum move distance: {MIN_MOVE_DISTANCE:.1f}")

def create_triangle(x, y, dx, dy, color):
    """Create a triangle pointing in the direction of movement"""
    base_size = 2 * TRIANGLE_SCALE
    angle = math.atan2(dy, dx)
    points = [
        (base_size * 0.5, 0),
        (-base_size * 0.5, -base_size * 0.3),
        (-base_size * 0.5, base_size * 0.3)
    ]
    return Polygon([(x + p[0] * math.cos(angle) - p[1] * math.sin(angle),
                     y + p[0] * math.sin(angle) + p[1] * math.cos(angle)) for p in points],
                   closed=True, color=color, zorder=3)

global current_step
current_step = 0

def update_frame(frame, grid_image, house_scatter, current_target_scatter, param_text, ax):
    #hi
    global tick, grid, turtles, triangle_artists, houses, order_parameters_history, calculation_started, history_index, current_step
    current_step+=1

    if paused:
        return (grid_image, *triangle_artists, house_scatter, current_target_scatter, param_text)

    tick += 1

    # Move and check for target reached for each turtle
    for turtle in turtles:
        move_turtle(turtle, tick)
        if turtle['is_moving_to_house'] and turtle['current_house_index'] != -1:
            house = houses[turtle['current_house_index']]
            if distance_between(turtle['x'], turtle['y'], house['x'], house['y']) < 1:
                turtle['houses_visited'].add(turtle['current_house_index'])
                set_next_house_destination(turtle)

    # Initialize grids for visualization
    route_grid = np.zeros((grid_height, grid_width))
    popularity_grid = np.zeros((grid_height, grid_width))

    # Apply popularity decay
    for y in range(grid_height):
        for x in range(grid_width):
            patch = grid[y][x]
            if tick - patch['last_visited'] > 10:
                decay_amount = patch['popularity'] * (POPULARITY_DECAY_RATE / 100.0)
                patch['popularity'] = max(0, patch['popularity'] - decay_amount)
                if patch['is_route'] and patch['popularity'] < MINIMUM_ROUTE_POPULARITY / 2:
                    patch['is_route'] = False
            route_grid[y, x] = 1 if patch['is_route'] else 0
            popularity_grid[y, x] = patch['popularity']

    # Prepare grid image
    grid_colors = np.zeros((grid_height, grid_width, 3))
    grid_colors[:, :] = BG_COLOR

    

    # Add popularity visualization
    if SHOW_POPULARITY:
        max_pop = max(MINIMUM_ROUTE_POPULARITY, np.max(popularity_grid)) if np.max(popularity_grid) > 0 else MINIMUM_ROUTE_POPULARITY
        norm_pop = np.minimum(1.0, popularity_grid / max_pop)
        for y in range(grid_height):
            for x in range(grid_width):
                if popularity_grid[y, x] > 0 and not grid[y][x]['is_route']:
                    green_intensity = min(1.0, 0.7 + (norm_pop[y, x] * 0.3))
                    red_blue_value = 0.2 + (norm_pop[y, x] * 0.2)
                    grid_colors[y, x] = (red_blue_value, green_intensity, red_blue_value)

    # Add route visualization
    for y in range(grid_height):
        for x in range(grid_width):
            if grid[y][x]['is_route']:
                grid_colors[y, x] = PATH_COLOR

    # Update grid image
    grid_image.set_array(grid_colors)

    # Update triangles (turtles)
    for artist in triangle_artists:
        if artist in ax.patches:
            artist.remove()
    triangle_artists = []
    for turtle in turtles:
        triangle = create_triangle(turtle['x'], turtle['y'], turtle['dx'], turtle['dy'], TURTLE_COLOR)
        ax.add_patch(triangle)
        triangle_artists.append(triangle)

    # Update house positions
    if houses:
        house_x = [house['x'] for house in houses]
        house_y = [house['y'] for house in houses]
        house_positions = np.column_stack((house_x, house_y))
        house_scatter.set_offsets(house_positions)
        house_scatter.set_visible(True)

        # Update current target positions
        current_targets = set()
        for turtle in turtles:
            if turtle['current_house_index'] != -1:
                current_targets.add(turtle['current_house_index'])

        if current_targets:
            target_x = [houses[idx]['x'] for idx in current_targets]
            target_y = [houses[idx]['y'] for idx in current_targets]
            target_positions = np.column_stack((target_x, target_y))
            current_target_scatter.set_offsets(target_positions)
            current_target_scatter.set_visible(True)
        else:
            current_target_scatter.set_visible(False)
    else:
        house_scatter.set_visible(False)
        current_target_scatter.set_visible(False)

    # Count turtles that have visited all houses
    turtles_completed = sum(1 for turtle in turtles if len(turtle['houses_visited']) == len(houses) and len(houses) > 0)

    # Average houses visited per turtle
    if len(houses) > 0:
        avg_houses_visited = sum(len(turtle['houses_visited']) for turtle in turtles) / max(1, len(turtles))
        avg_percent = (avg_houses_visited / len(houses)) * 100
    else:
        avg_percent = 0

    # Start calculating order parameters only after two houses are present
    if len(houses) >= 2:
        global calculation_started, order_parameters_history, history_index
        calculation_started = True
        popularity_entropy = calculate_popularity_entropy(grid)
        connectivity_index = calculate_connectivity_index(grid, grid_width, grid_height)
        path_density = calculate_path_density(grid, grid_width, grid_height)

        if order_parameters_history is None:
            max_steps = 5000  # Adjust as needed
            order_parameters_history = np.zeros((max_steps, 3))
            history_index = 0

        if history_index < order_parameters_history.shape[0]:
            order_parameters_history[history_index, 0] = popularity_entropy
            order_parameters_history[history_index, 1] = connectivity_index
            order_parameters_history[history_index, 2] = path_density
            history_index += 1
        else:
            order_parameters_history = np.vstack((order_parameters_history, np.zeros((order_parameters_history.shape[0], 3))))
            order_parameters_history[history_index, 0] = popularity_entropy
            order_parameters_history[history_index, 1] = connectivity_index
            order_parameters_history[history_index, 2] = path_density
            history_index += 1

    elif len(houses) < 2:
        calculation_started = False
        order_parameters_history = None
        history_index = 0

    # Update parameter text
    entropy_val = f"{order_parameters_history[history_index - 1, 0]:.2f}" if calculation_started and order_parameters_history is not None and history_index > 0 else "N/A"
    connectivity_val = f"{order_parameters_history[history_index - 1, 1]:.2f}" if calculation_started and order_parameters_history is not None and history_index > 0 else "N/A"
    density_val = f"{order_parameters_history[history_index - 1, 2]:.2f}" if calculation_started and order_parameters_history is not None and history_index > 0 else "N/A"

    timestep_display = f"Recording Timestep: {history_index}" if calculation_started else "Recording Timestep: N/A (Waiting for 2 houses)"

    param_text_str = "\n".join([
        f"Turtles: {WALKER_COUNT} (↑/↓ to change)",
        f"Path Influence: {PATH_INFLUENCE:.1f} (+/- to change)",
        f"Min Move Dist: {MIN_MOVE_DISTANCE:.1f} (</> to change)",
        f"Decay Rate: {POPULARITY_DECAY_RATE}",
        f"Houses: {len(houses)} (Click to add/remove)",
        f"Turtles completed: {turtles_completed}/{WALKER_COUNT}",
        f"Avg houses visited: {avg_percent:.1f}%",
        f"Popularity Entropy: {entropy_val}",
        f"Connectivity Index: {connectivity_val}",
        f"Path Density: {density_val}",
        f"Houses added at ticks: {house_addition_time_steps}"
        f"Space: Pause/Resume",
        f"R: Reset",
        f"C: Clear Houses"
    ])
    param_text.set_text(param_text_str)

    return (grid_image, *triangle_artists, house_scatter, current_target_scatter, param_text)

# -----------------------------------------------------------------
import math
import numpy as np
from collections import Counter

# Global variable to store order parameters as a NumPy array
order_parameters_history = None
calculation_started = False
history_index = 0

def calculate_popularity_entropy(grid):
    """Calculates the Shannon entropy of path popularity."""
    popularities = [patch['popularity'] for row in grid for patch in row]
    if not popularities:
        return 0
    counts = Counter(popularities)
    total = len(popularities)
    entropy = 0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

def calculate_connectivity_index(grid, grid_width, grid_height):
    """Calculates the average connectivity of established paths."""
    established_path_count = 0
    total_connectivity = 0
    for y in range(grid_height):
        for x in range(grid_width):
            if grid[y][x]['is_route']:
                established_path_count += 1
                neighbors = 0
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # Moore neighborhood (no diagonals)
                    nx = (x + dx) % grid_width
                    ny = (y + dy) % grid_height
                    if grid[ny][nx]['is_route']:
                        neighbors += 1
                total_connectivity += neighbors
    if established_path_count > 0:
        return total_connectivity / established_path_count
    else:
        return 0

def calculate_path_density(grid, grid_width, grid_height):
    """Calculates the density of established paths."""
    established_count = sum(1 for row in grid for patch in row if patch['is_route'])
    total_patches = grid_width * grid_height
    return established_count / total_patches if total_patches > 0 else 0

def update_frame(frame, grid_image, house_scatter, current_target_scatter, param_text, ax):
    global tick, grid, turtles, triangle_artists, houses, order_parameters_history, calculation_started, history_index, current_step

    if paused:
        return (grid_image, *triangle_artists, house_scatter, current_target_scatter, param_text)

    tick += 1

    # Move and check for target reached for each turtle
    for turtle in turtles:
        move_turtle(turtle, tick)
        if turtle['is_moving_to_house'] and turtle['current_house_index'] != -1:
            house = houses[turtle['current_house_index']]
            if distance_between(turtle['x'], turtle['y'], house['x'], house['y']) < 1:
                turtle['houses_visited'].add(turtle['current_house_index'])
                set_next_house_destination(turtle)

    # Initialize grids for visualization
    route_grid = np.zeros((grid_height, grid_width))
    popularity_grid = np.zeros((grid_height, grid_width))

    # Apply popularity decay
    for y in range(grid_height):
        for x in range(grid_width):
            patch = grid[y][x]
            if tick - patch['last_visited'] > 10:
                decay_amount = patch['popularity'] * (POPULARITY_DECAY_RATE / 100.0)
                patch['popularity'] = max(0, patch['popularity'] - decay_amount)
                if patch['is_route'] and patch['popularity'] < MINIMUM_ROUTE_POPULARITY / 2:
                    patch['is_route'] = False
            route_grid[y, x] = 1 if patch['is_route'] else 0
            popularity_grid[y, x] = patch['popularity']

    # Prepare grid image
    grid_colors = np.zeros((grid_height, grid_width, 3))
    grid_colors[:, :] = BG_COLOR

    # Add popularity visualization
    if SHOW_POPULARITY:
        max_pop = max(MINIMUM_ROUTE_POPULARITY, np.max(popularity_grid)) if np.max(popularity_grid) > 0 else MINIMUM_ROUTE_POPULARITY
        norm_pop = np.minimum(1.0, popularity_grid / max_pop)
        for y in range(grid_height):
            for x in range(grid_width):
                if popularity_grid[y, x] > 0 and not grid[y][x]['is_route']:
                    green_intensity = min(1.0, 0.7 + (norm_pop[y, x] * 0.3))
                    red_blue_value = 0.2 + (norm_pop[y, x] * 0.2)
                    grid_colors[y, x] = (red_blue_value, green_intensity, red_blue_value)

    # Add route visualization
    for y in range(grid_height):
        for x in range(grid_width):
            if grid[y][x]['is_route']:
                grid_colors[y, x] = PATH_COLOR

    # Update grid image
    grid_image.set_array(grid_colors)

    # Update triangles (turtles)
    for artist in triangle_artists:
        if artist in ax.patches:
            artist.remove()
    triangle_artists = []
    for turtle in turtles:
        triangle = create_triangle(turtle['x'], turtle['y'], turtle['dx'], turtle['dy'], TURTLE_COLOR)
        ax.add_patch(triangle)
        triangle_artists.append(triangle)

    # Update house positions
    if houses:
        house_x = [house['x'] for house in houses]
        house_y = [house['y'] for house in houses]
        house_positions = np.column_stack((house_x, house_y))
        house_scatter.set_offsets(house_positions)
        house_scatter.set_visible(True)

        # Update current target positions
        current_targets = set()
        for turtle in turtles:
            if turtle['current_house_index'] != -1:
                current_targets.add(turtle['current_house_index'])

        if current_targets:
            target_x = [houses[idx]['x'] for idx in current_targets]
            target_y = [houses[idx]['y'] for idx in current_targets]
            target_positions = np.column_stack((target_x, target_y))
            current_target_scatter.set_offsets(target_positions)
            current_target_scatter.set_visible(True)
        else:
            current_target_scatter.set_visible(False)
    else:
        house_scatter.set_visible(False)
        current_target_scatter.set_visible(False)

    # Count turtles that have visited all houses
    turtles_completed = sum(1 for turtle in turtles if len(turtle['houses_visited']) == len(houses) and len(houses) > 0)

    # Average houses visited per turtle
    if len(houses) > 0:
        avg_houses_visited = sum(len(turtle['houses_visited']) for turtle in turtles) / max(1, len(turtles))
        avg_percent = (avg_houses_visited / len(houses)) * 100
    else:
        avg_percent = 0

    # Start calculating order parameters only after two houses are present
    if len(houses) >= 2:
        global calculation_started, order_parameters_history, history_index
        calculation_started = True
        popularity_entropy = calculate_popularity_entropy(grid)
        connectivity_index = calculate_connectivity_index(grid, grid_width, grid_height)
        path_density = calculate_path_density(grid, grid_width, grid_height)

        if order_parameters_history is None:
            # Initialize the NumPy array with an estimated maximum number of steps
            max_steps = 5000  # Adjust as needed
            order_parameters_history = np.zeros((max_steps, 3))
            history_index = 0

        if history_index < order_parameters_history.shape[0]:
            order_parameters_history[history_index, 0] = popularity_entropy
            order_parameters_history[history_index, 1] = connectivity_index
            order_parameters_history[history_index, 2] = path_density
            history_index += 1
        else:
            # Resize the array if the estimated maximum is reached
            order_parameters_history = np.vstack((order_parameters_history, np.zeros((order_parameters_history.shape[0], 3))))
            order_parameters_history[history_index, 0] = popularity_entropy
            order_parameters_history[history_index, 1] = connectivity_index
            order_parameters_history[history_index, 2] = path_density
            history_index += 1

    elif len(houses) < 2:
        calculation_started = False
        order_parameters_history = None
        history_index = 0

    # Update parameter text
    entropy_val = f"{order_parameters_history[history_index - 1, 0]:.2f}" if calculation_started and order_parameters_history is not None and history_index > 0 else "N/A"
    connectivity_val = f"{order_parameters_history[history_index - 1, 1]:.2f}" if calculation_started and order_parameters_history is not None and history_index > 0 else "N/A"
    density_val = f"{order_parameters_history[history_index - 1, 2]:.2f}" if calculation_started and order_parameters_history is not None and history_index > 0 else "N/A"

    param_text_str = "\n".join([
        f"Turtles: {WALKER_COUNT} (↑/↓ to change)",
        f"Path Influence: {PATH_INFLUENCE:.1f} (+/- to change)",
        f"Min Move Dist: {MIN_MOVE_DISTANCE:.1f} (</> to change)",
        f"Decay Rate: {POPULARITY_DECAY_RATE}",
        f"Houses: {len(houses)} (Click to add/remove)",
        f"Turtles completed: {turtles_completed}/{WALKER_COUNT}",
        f"Avg houses visited: {avg_percent:.1f}%",
        f"Popularity Entropy: {entropy_val}",
        f"Connectivity Index: {connectivity_val}",
        f"Path Density: {density_val}",
        f"Space: Pause/Resume",
        f"R: Reset",
        f"C: Clear Houses"
    ])
    param_text.set_text(param_text_str)

    return (grid_image, *triangle_artists, house_scatter, current_target_scatter, param_text)

# -----------------------------------------------------------------
def main():
    global grid_width, grid_height, houses, order_parameters_history

    grid_width, grid_height = WIDTH, HEIGHT
    houses = []
    order_parameters_history = None # Initialize as None

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(left=0.1, bottom=0.15)

    # Initialize simulation
    reset_simulation()

    # Set up axes
    ax.set_xlim(0, grid_width)
    ax.set_ylim(0, grid_height)
    ax.set_title('Emergent Path Formation Simulation')

    # Create initial grid image
    grid_colors = np.zeros((grid_height, grid_width, 3))
    grid_colors[:, :] = BG_COLOR
    grid_image = ax.imshow(grid_colors, origin='lower')

    # Create scatter plots for houses and current targets
    house_scatter = ax.scatter([], [], marker='s', color=HOUSE_COLOR, s=40, zorder=2)
    current_target_scatter = ax.scatter([], [], marker='*', color=CURRENT_TARGET_COLOR, s=100, zorder=1)

    # Create parameter text
    param_text = ax.text(0.02, 0.02, "", transform=ax.transAxes,
                         verticalalignment='bottom', bbox=dict(boxstyle='round',
                         facecolor='white', alpha=0.7))

    # Create buttons
    reset_button_ax = plt.axes([0.8, 0.05, 0.15, 0.04])
    reset_button = Button(reset_button_ax, 'Reset')
    reset_button.on_clicked(reset_simulation)

    pause_button_ax = plt.axes([0.8, 0.1, 0.15, 0.04])
    pause_button = Button(pause_button_ax, 'Pause/Resume')
    pause_button.on_clicked(toggle_pause)

    clear_houses_button_ax = plt.axes([0.8, 0.15, 0.15, 0.04])
    clear_houses_button = Button(clear_houses_button_ax, 'Clear Houses')
    clear_houses_button.on_clicked(clear_houses)

    # Connect events
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Start animation
    ani = FuncAnimation(fig, update_frame, fargs=(grid_image, house_scatter, current_target_scatter, param_text, ax), interval=50, blit=True)

    plt.show()

    # After the animation is closed, you can access the order parameter history:
    # if order_parameters_history is not None and history_index > 1:
    #     final_history = order_parameters_history[:history_index]
    #     if final_history.shape[0] > 1:
    #         covariance_matrix = np.cov(final_history, rowvar=False)
    #         correlation_matrix = np.corrcoef(final_history, rowvar=False)
    #         print("\nOrder Parameter History (NumPy Array):")
    #         print(final_history)
    #         print("\nCovariance Matrix:")
    #         print(covariance_matrix)
    #         print("\nCorrelation Matrix:")
    #         print(correlation_matrix)
    #     else:
    #         print("\nNot enough time steps with at least two houses to calculate covariance and correlation.")
    # else:
    #     print("\nNo order parameter data collected (less than two houses added).")

    if order_parameters_history is not None and order_parameters_history.shape[0] > 1:
        final_history = order_parameters_history[:history_index]
        mean = np.mean(final_history, axis=0)
        std = np.std(final_history, axis=0)
        global normalized_history
        normalized_history = (final_history - mean) / std

        # covariance_matrix = np.cov(normalized_history, rowvar=False)
        # correlation_matrix = np.corrcoef(normalized_history, rowvar=False)

        print("\nNormalized Order Parameter History:")
        print(normalized_history)
        # print("\nCovariance Matrix (of normalized data):")
        # print(covariance_matrix)
        # print("\nCorrelation Matrix (of normalized data):")
        # print(correlation_matrix)
    else:
        print("\nNot enough data to normalize.")

if __name__ == "__main__":
    main()
     # After the animation is closed, you can access the order parameter history:
    if order_parameters_history is not None and history_index > 0:
        final_history = order_parameters_history[:history_index]
        print(f"\nShape of final_history matrix (actual recorded data): {final_history.shape}")
        t = house_addition_time_steps
        print("timesteps at which houses are added: ", t)
        if final_history.shape[0] > 1:
            covariance_matrix = np.cov(final_history, rowvar=False)
            correlation_matrix = np.corrcoef(final_history, rowvar=False)
            print("\nOrder Parameter History (NumPy Array):")
            print(final_history)
            print("\nCovariance Matrix:")
            print(covariance_matrix)
            print("\nCorrelation Matrix:")
            print(correlation_matrix)
        else:
            print("\nNot enough time steps with at least two houses to calculate covariance and correlation.")
    else:
        print("\nNo order parameter data collected (less than two houses added).")

#----------------------------------------------------------------------------------------------------------------------
from sklearn.decomposition import PCA
import numpy as np

pca = PCA(n_components=3)
pca.fit(normalized_history)

# PC1 component loadings (weights for each order parameter)
pc1_weights = pca.components_[0]

order_params = ["Popularity Entropy", "Connectivity_Index", "Path Density"]

# Rank parameters by absolute weight in PC1
ranking = sorted(zip(order_params, pc1_weights), key=lambda x: abs(x[1]), reverse=True)

# Print ranking
print("PCA Ranking of Order Parameters (by importance in PC1):")
for name, weight in ranking:
    print(f"{name}: weight = {weight:.4f}")
# ----------------------------------------------------------------------------
# create plots
labels = ["Popularity Entropy", "Connectivity Index", "Path Density"]
num_snapshots = normalized_history.shape[0]
time = np.arange(num_snapshots)

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharex=True)

for i in range(3):
    axes[i].plot(time, normalized_history[:, i], label=labels[i], color='C' + str(i))
    axes[i].set_title(labels[i])
    axes[i].set_xlabel("Time Step")
    axes[i].set_ylabel("Normalized Value")
    axes[i].grid(True)

    for j in t:
        axes[i].axvline(x=j, color='red', linestyle='--', linewidth=1, alpha=0.7)

plt.tight_layout()
plt.show()  
# ----------------------------------------------------------------------------
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=2)
compressed = svd.fit_transform(normalized_history)

plt.plot(compressed[:, 0], label='SVD PC1')
plt.plot(compressed[:, 1], label='SVD PC2')
plt.legend()
plt.title("Compressed Dynamics from SVD")
plt.xlabel("Time Step")
plt.grid(True)
plt.show()
# -----------------------------------------------------------------------------

import numpy as np
from pysr import PySRRegressor
import matplotlib.pyplot as plt

# Create time input and targets from compressed data
timesteps = compressed.shape[0]
X = np.arange(timesteps).reshape(-1, 1)  # time as input
y1 = compressed[:, 0]  # SVD PC1
y2 = compressed[:, 1]  # SVD PC2

# Configure PySR model
model = PySRRegressor(
    niterations=100,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "exp", "log"],
    model_selection="best",  # pick simplest accurate model
    loss="loss(x, y) = (x - y)^2",
    maxsize=20,
    verbosity=1,
)

# Fit for PC1
model.fit(X, y1)
best_pc1_eq = model.get_best()
print("\n Best equation for PC1:\n", best_pc1_eq)

# Predict and plot PC1
pred1 = model.predict(X)
plt.plot(y1, label='True PC1')
plt.plot(pred1, label='Predicted PC1', linestyle='--')
plt.title("SVD PC1 - PySR Prediction")
plt.legend()
plt.grid(True)
plt.show()

# Fit for PC2
model.fit(X, y2)
best_pc2_eq = model.get_best()
print("\n Best equation for PC2:\n", best_pc2_eq)

# Predict and plot PC2
pred2 = model.predict(X)
plt.plot(y2, label='True PC2')
plt.plot(pred2, label='Predicted PC2', linestyle='--')
plt.title("SVD PC2 - PySR Prediction")
plt.legend()
plt.grid(True)
plt.show() 
