import numpy as np
import cupy as cp  # CUDA-accelerated NumPy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the scaling factor and angle of rotation
k = 0.5
theta = np.pi / 2  # 45 degrees

# Define the vector field formula


def vector_field(x, y):
    u = -k * y * np.cos(theta) - k * x * np.sin(theta)
    v = -k * x * np.cos(theta) + k * y * np.sin(theta)
    return u, v


def vector_field_curly(x, y):
    u = -k * y * np.cos(theta) + k * x * np.sin(theta)
    v = k * x * np.cos(theta) + k * y * np.sin(theta)
    return u, v


def vector_field_away(x, y):
    u = -k * y * np.cos(theta) - k * x * np.sin(theta)
    v = -k * x * np.cos(theta) + k * y * np.sin(theta)
    return u, v


def vector_field_drain(x, y):
    u = y - x
    v = -x - y
    return u, v


def vector_field_collide(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    u = -r*np.cos(theta)
    v = -r*np.sin(theta)
    return u, v


def vector_field_gravity(x, y):
    gravity_constant = 0.5  # Adjust the gravity constant as needed
    origin_x, origin_y = 0, 0  # Origin point

    distance = np.sqrt((x - origin_x)**2 + (y - origin_y)**2)
    u = y - x - gravity_constant * (x - origin_x) / distance**3
    v = -x - y - gravity_constant * (y - origin_y) / distance**3
    return u, v


def vector_field_damped(x, y):
    gravity_constant = 0.1  # Adjust the gravity constant as needed
    damping_factor = 0.5  # Adjust the damping factor as needed
    origin_x, origin_y = 0, 0  # Origin point

    distance = np.sqrt((x - origin_x)**2 + (y - origin_y)**2)
    u = y - x - gravity_constant * \
        (x - origin_x) / distance**3 - damping_factor * y
    v = -x - y - gravity_constant * \
        (y - origin_y) / distance**3 + damping_factor * x

    return u, v


# Define the particle-particle collision function


def collide2(p1, p2, v1, v2):
    m1, m2 = 1, 1  # Assume equal masses for simplicity
    d = p2 - p1
    d_norm = np.linalg.norm(d, axis=1)
    mask = d_norm < 0.2  # Adjust the collision threshold as needed
    n = d / d_norm[:, None]
    v_rel = v2 - v1
    v_rel_norm = np.einsum('ij,ij->i', v_rel, n)
    v_rel_coll = np.where(v_rel_norm < 0, -v_rel_norm[:, None] * n, 0)
    v1 += np.sum(np.where(mask[:, None], v_rel_coll / m1, 0), axis=0)
    v2 -= np.sum(np.where(mask[:, None], v_rel_coll / m2, 0), axis=0)
    return v1, v2


def collide1(p1, p2, v1, v2):
    m1, m2 = 1, 1  # Assume equal masses for simplicity
    d = p2 - p1
    d_norm = np.linalg.norm(d, axis=1)
    mask = d_norm < 0.2  # Adjust the collision threshold as needed
    n = d / d_norm[:, None]
    v_rel = v2 - v1
    v_rel_norm = np.einsum('ij,ij->i', v_rel, n)
    v_rel_coll = np.where(v_rel_norm < 0, -v_rel_norm[:, None] * n, 0)
    v1 += np.sum(np.where(mask[:, None], v_rel_coll,
                 0) / m1, axis=0, keepdims=True)
    v2 -= np.sum(np.where(mask[:, None], v_rel_coll,
                 0) / m2, axis=0, keepdims=True)
    return v1, v2


def collide(p1, p2, v1, v2):
    m1, m2 = 1, 1  # Assume equal masses for simplicity
    d = p2 - p1
    d_norm = np.linalg.norm(d, axis=1)
    mask = d_norm < 0.2  # Adjust the collision threshold as needed
    n = d / d_norm[:, None]
    v_rel = v2 - v1
    v_rel_norm = np.einsum('ij,ij->i', v_rel, n)
    v_rel_norm = v_rel_norm[:, np.newaxis]  # Reshape to match n's shape
    v_rel_coll = np.where(v_rel_norm < 0, -v_rel_norm * n, 0)
    v1 += np.sum(np.where(mask[:, None], v_rel_coll / m1, 0), axis=0)
    v2 -= np.sum(np.where(mask[:, None], v_rel_coll / m2, 0), axis=0)
    return v1, v2


# Define the drag effect function


def apply_drag1(v, distance):
    drag_factor = 0.2  # Adjust the drag factor as needed
    v_norm = np.linalg.norm(v, axis=1)
    drag = drag_factor * v_norm * distance
    v -= (v / v_norm[:, None]) * drag[:, None]
    return v


def apply_drag2(v, distance):
    drag_factor = 0.2  # Adjust the drag factor as needed
    v_norm = np.linalg.norm(v, axis=1)
    drag = drag_factor * v_norm * distance
    # Added minimum threshold of 0.01
    v -= (v / v_norm[:, None]) * np.maximum(drag[:, None], 0.05)
    return v


def apply_drag3(v, distance):
    drag_factor = 0.2  # Adjust the drag factor as needed
    damping_factor = 0.1  # Adjust the damping factor as needed

    v_norm = np.linalg.norm(v, axis=1)
    drag = drag_factor * v_norm * distance

    # Compute the angular velocity
    angular_velocity = v[:, 0] * v[:, 1] / v_norm

    # Apply damping to the angular velocity
    damping = damping_factor * angular_velocity
    # Apply damping only when velocity is low
    damping = np.where(v_norm < 0.1, damping, 0)

    # Apply the drag force while preserving the angular velocity
    damping_matrix = np.column_stack((damping, -damping))
    v -= (v / v_norm[:, None]) * np.maximum(drag[:, None],
                                            0.05) + damping_matrix * distance[:, None]
    return v

# Define the drag effect function


def apply_drag(v, distance):
    drag_factor = 0.2  # Adjust the drag factor as needed
    damping_factor = 0.1  # Adjust the damping factor as needed

    v_norm = np.linalg.norm(v, axis=1)
    drag = drag_factor * v_norm * distance

    # Compute the angular velocity
    angular_velocity = v[:, 0] * v[:, 1] / v_norm

    # Apply damping to the angular velocity
    damping = damping_factor * angular_velocity
    # Apply damping only when velocity is low
    damping = np.where(v_norm < 0.1, damping, 0)

    # Apply the drag force while preserving the angular velocity
    damping_matrix = np.column_stack((damping, -damping))
    v -= (v / v_norm[:, None]) * np.maximum(drag[:, None],
                                            0.05) + damping_matrix * distance[:, None]

    return v


# Initialize the number of particles
num_particles = 100

# Initialize the particles' positions and velocities
# Random positions in extended range
positions = np.random.uniform(-10, 10, (num_particles, 2))
velocities = np.random.uniform(-0.1, 0.1, (num_particles, 2))

# Create a figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)  # Adjusted x-axis limits
ax.set_ylim(-10, 10)  # Adjusted y-axis limits

# Create a quiver plot of the vector field
x = np.linspace(-10, 10, 20)  # Adjusted x-axis values
y = np.linspace(-10, 10, 20)  # Adjusted y-axis values
X, Y = np.meshgrid(x, y)
U, V = vector_field_damped(X, Y)
ax.quiver(X, Y, U, V)

# Create scatter plot for the particles
particles = ax.scatter([], [], color='red')

# Update function to compute particles' new positions


def update(frame):
    global positions, velocities
    dt = 0.1  # Time step

    # Compute new positions based on the vector field using GPU acceleration
    with cp.cuda.Device(0):
        positions_gpu = cp.asarray(positions)
        velocities_gpu = cp.asarray(velocities)

        u_gpu, v_gpu = vector_field_damped(
            positions_gpu[:, 0], positions_gpu[:, 1])

        positions_gpu += velocities_gpu * dt
        velocities_gpu += cp.column_stack((u_gpu, v_gpu)) * dt

        positions = cp.asnumpy(positions_gpu)
        velocities = cp.asnumpy(velocities_gpu)

    # Apply drag effect based on the distance traveled
    distances = np.linalg.norm(velocities, axis=1) * dt
    velocities = apply_drag(velocities, distances)

    # Check for collisions between particles
    for i in range(num_particles):
        distances = np.linalg.norm(positions[i] - positions, axis=1)
        # Adjust the collision threshold as needed
        collisions = (distances < 0.2) & (distances > 0)
        if np.any(collisions):
            collided_indices = np.where(collisions)[0]
            v1, v2 = velocities[i:i+1], velocities[collided_indices]
            velocities[i:i+1], velocities[collided_indices] = collide(
                positions[i:i+1], positions[collided_indices], v1, v2)

    # Update particles' positions in the scatter plot
    particles.set_offsets(positions)


# Create the animation using FuncAnimation
animation = FuncAnimation(fig, update, frames=100, interval=50, blit=False)

# Show the animation
plt.show()
