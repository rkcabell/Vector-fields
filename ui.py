import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import itertools
import vector_fields as vf

# Define the scaling factor and angle of rotation
k = 0.5
theta = np.pi / 4  # 45 degrees
# Initialize the number of particles
num_particles = 200
preserve_angular_momentum_enabled = False
orbit_center_enabled = True
collision_enabled = False
selected_function = "Simple"
simple_vectorfields = ["Simple", "Drain", "Collide", "Gravity", "Damped"]
complex_vectorfields = ["Curly", "Away"]
# Define the particle-particle collision function


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
def apply_drag(v, distance):
    drag_factor = 0.2  # Adjust the drag factor as needed
    damping_factor = 0.1  # Adjust the damping factor as needed

    v_norm = np.linalg.norm(v, axis=1)
    drag = drag_factor * v_norm * distance

    # Apply the drag force
    v -= (v / v_norm[:, None]) * np.maximum(drag[:, None], 0.05)

    # Apply damping to the velocity
    damping = damping_factor * v_norm
    damping = np.where(v_norm < 0.1, damping, 0)
    v -= (v / v_norm[:, None]) * damping[:, None]

    return v


def apply_drag_preserve_angular(v, distance):

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


# Initialize the particles' positions and velocities
# Random positions in extended range
positions = np.random.uniform(-10, 10, (num_particles, 2))
velocities = np.random.uniform(-0.1, 0.1, (num_particles, 2))


def calculate_particle_velocities(positions, U, V):
    indices = ((positions - x[0]) / (x[1] - x[0])).astype(int)
    indices = np.clip(indices, 0, U.shape[0] - 1)

    velocities = k * \
        np.column_stack((U[indices[:, 1], indices[:, 0]],
                        V[indices[:, 1], indices[:, 0]]))
    return velocities
# Create a function to update the instance variable values


def update_instance_variables():
    global k, theta
    k = k_value.get()
    theta = theta_value.get()


# Create the Tkinter UI
root = tk.Tk()
root.title("Vector Field Simulation")

# Define a callback function to handle window close event


def on_closing():
    root.destroy()  # Close the tkinter window
    plt.close()  # Close the matplotlib figure


# Set the callback function for the window close event
root.protocol("WM_DELETE_WINDOW", on_closing)

# Create a frame for the vector field options
# Set the default value to "Gravity"
vector_field_var = tk.StringVar(value="Gravity")
vector_field_frame = ttk.LabelFrame(root, text="Vector Field")
vector_field_frame.pack(padx=10, pady=10)

# Create a variable to store the selected vector field function
selected_vector_field = tk.StringVar(value="Gravity")

# Create a function to update the vector field based on the selected option
this_vfield = vf.VectorField(vf.gravity)


def evaluate_vector_field(selected_function):
    global this_vfield
    try:
        if selected_function in complex_vectorfields:
            this_vfield.evaluate(X, Y, k, theta)
        elif selected_function in simple_vectorfields:
            this_vfield.evaluate(X, Y)
        else:
            raise ValueError("Invalid selected function")
    except Exception as e:
        raise Exception("Failed to evaluate vector field:", str(e))

    return this_vfield.U, this_vfield.V


def update_vector_field():
    global selected_function, U, V, velocities, this_vfield

    selected_function = selected_vector_field.get()
    if selected_function == "Curly":
        this_vfield = vf.VectorField(vf.curly)
    elif selected_function == "Away":
        this_vfield = vf.VectorField(vf.away)
    elif selected_function == "Drain":
        this_vfield = vf.VectorField(vf.drain)
    elif selected_function == "Collide":
        this_vfield = vf.VectorField(vf.collide)
    elif selected_function == "Gravity":
        this_vfield = vf.VectorField(vf.gravity)
    elif selected_function == "Damped":
        this_vfield = vf.VectorField(vf.damped)
    else:  # Default to Simple
        this_vfield = vf.VectorField(vf.simple)

    U, V = evaluate_vector_field(selected_function)

    # Update the quiver plot
    quiver.set_UVC(U, V)

    # Update particle velocities based on the new field
    velocities = calculate_particle_velocities(positions, U, V)


def toggle_angular_momentum():
    global preserve_angular_momentum_enabled
    preserve_angular_momentum_enabled = not preserve_angular_momentum_enabled


def toggle_collision():
    global collision_enabled
    collision_enabled = not collision_enabled


def toggle_orbit():
    global orbit_center_enabled
    orbit_center_enabled = not orbit_center_enabled


def button_collision():
    toggle_collision()
    if collision_enabled:
        btn_collision.config(text="Collision: ON")
    else:
        btn_collision.config(text="Collision: OFF")


def button_PAM():
    toggle_angular_momentum()
    if preserve_angular_momentum_enabled:
        btn_PAM.config(text="Preserve Angular Momentum: ON")
    else:
        btn_PAM.config(text="Preserve Angular Momentum: OFF")


def button_orbit():
    toggle_orbit()
    if orbit_center_enabled:
        btn_orbit.config(text="Orbit center: ON")
    else:
        btn_orbit.config(text="Orbit center: OFF")


vars_frame = ttk.LabelFrame(root, text="Variables")
vars_frame.pack(padx=10, pady=10)

btn_PAM = ttk.Button(
    vars_frame, text="Preserve Angular Momentum: " + str(preserve_angular_momentum_enabled), command=button_PAM)
btn_PAM.pack(padx=10, pady=5)

btn_collision = ttk.Button(
    vars_frame, text="Collision: " + str(collision_enabled), command=button_collision)
btn_collision.pack(padx=10, pady=5)

btn_orbit = ttk.Button(
    vars_frame, text="Orbit Center: " + str(orbit_center_enabled), command=button_orbit)
btn_orbit.pack(padx=10, pady=5)


# Create a radio button for each vector field option
vector_field_options = ["Damped", "Curly",
                        "Away", "Drain", "Collide", "Gravity"]
for option in vector_field_options:
    ttk.Radiobutton(vector_field_frame, text=option, variable=selected_vector_field,
                    value=option, command=update_vector_field).pack(anchor=tk.W)

# Create a frame for the instance variable options
instance_vars_frame = ttk.LabelFrame(root, text="Instance Variables")
instance_vars_frame.pack(padx=10, pady=10)

# Create variables to store the instance variable values
k_value = tk.DoubleVar(value=k)
theta_value = tk.DoubleVar(value=theta)


# Create labels and entry fields for the instance variables
ttk.Label(instance_vars_frame, text="Vector field scaling factor:").grid(
    column=0, row=0, sticky=tk.E)
ttk.Label(instance_vars_frame, text="Angle of Rotation:").grid(
    column=0, row=1, sticky=tk.E)
ttk.Entry(instance_vars_frame, textvariable=k_value,
          width=10).grid(column=1, row=0, sticky=tk.W)
ttk.Entry(instance_vars_frame, textvariable=theta_value,
          width=10).grid(column=1, row=1, sticky=tk.W)

# Create a button to apply the instance variable changes
ttk.Button(instance_vars_frame, text="Apply",
           command=update_instance_variables).grid(column=1, row=2)

# Create a figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)  # Adjusted x-axis limits
ax.set_ylim(-10, 10)  # Adjusted y-axis limits

# Create a quiver plot of the vector field
x = np.linspace(-10, 10, 20)  # Adjusted x-axis values
y = np.linspace(-10, 10, 20)  # Adjusted y-axis values
X, Y = np.meshgrid(x, y)
U, V = evaluate_vector_field(selected_function)
quiver = ax.quiver(X, Y, U, V)

# Create scatter plot for the particles
particles = ax.scatter([], [], color='red')


def update(frame):
    global positions, velocities
    dt = 0.1  # Time step
    results = []

    # Compute new positions based on the vector field using multithreading
    with ThreadPoolExecutor() as executor:
        if selected_function in simple_vectorfields:
            results = list(executor.map(this_vfield.evaluate,
                                        positions[:, 0], positions[:, 1]))
        elif selected_function in complex_vectorfields:
            results = list(executor.map(this_vfield.evaluate,
                                        positions[:, 0], positions[:, 1], itertools.repeat(k), itertools.repeat(theta)))

    u = np.array([result[0] for result in results])
    v = np.array([result[1] for result in results])

    new_positions = positions + velocities * dt
    new_velocities = velocities + np.column_stack((u, v)) * dt

    # Apply drag effect based on the distance traveled
    distances = np.linalg.norm(new_velocities, axis=1) * dt
    if orbit_center_enabled:
        if preserve_angular_momentum_enabled == True:
            new_velocities = apply_drag_preserve_angular(
                new_velocities, distances)
        else:
            new_velocities = apply_drag(new_velocities, distances)

    # Check for collisions between particles
    if collision_enabled:
        for i in range(num_particles):
            distances = np.linalg.norm(
                new_positions[i] - new_positions, axis=1)
            # Adjust the collision threshold as needed
            collisions = (distances < 0.2) & (distances > 0)
            if np.any(collisions):
                collided_indices = np.where(collisions)[0]
                v1, v2 = new_velocities[i:i +
                                        1], new_velocities[collided_indices]
                new_velocities[i:i+1], new_velocities[collided_indices] = collide(
                    new_positions[i:i+1], new_positions[collided_indices], v1, v2)

    # Update particle positions and velocities
    positions = new_positions
    velocities = new_velocities

    # Update particles' positions in the scatter plot
    particles.set_offsets(positions)


# Create the animation using FuncAnimation
animation = FuncAnimation(fig, update, frames=100, interval=50, blit=False)

# Create a tkinter canvas to embed the matplotlib graph
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Start the Tkinter main loop
root.mainloop()
