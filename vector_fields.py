import numpy as np


class VectorField:

    def __init__(self):
        self.function = gravity
        self.U = None
        self.V = None

    def __init__(self, function):
        self.function = function
        self.U = None
        self.V = None

    def evaluate(self, *args):
        self.U, self.V = self.function(*args)
        return self.U, self.V

# Define the vector field formula


def simple(x, y):
    u = y - x
    v = -x - y
    return u, v


def curly(x, y, k, theta):
    u = -k * y * np.cos(theta) + k * x * np.sin(theta)
    v = k * x * np.cos(theta) + k * y * np.sin(theta)
    return u, v


def away(x, y, k, theta):
    u = -k * y * np.cos(theta) - k * x * np.sin(theta)
    v = -k * x * np.cos(theta) + k * y * np.sin(theta)
    return u, v


def drain(x, y):
    u = y - x
    v = -x - y
    return u, v


def collide(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    u = -r*np.cos(theta)
    v = -r*np.sin(theta)
    return u, v


def gravity(x, y):
    gravity_constant = 0.5  # Adjust the gravity constant as needed
    origin_x, origin_y = 0, 0  # Origin point

    distance = np.sqrt((x - origin_x)**2 + (y - origin_y)**2)
    u = y - x - gravity_constant * (x - origin_x) / distance**3
    v = -x - y - gravity_constant * (y - origin_y) / distance**3
    return u, v


def damped(x, y):
    gravity_constant = 0.1  # Adjust the gravity constant as needed
    damping_factor = 0.5  # Adjust the damping factor as needed
    epsilon = 1e-6  # Small positive value for stability
    origin_x, origin_y = 0, 0  # Origin point
    min_distance = 1e-10  # Set a minimum distance value

    distance = np.sqrt((x - origin_x)**2 + (y - origin_y)**2)
    scaled_distance = np.maximum(distance, min_distance)
    scaled_distance_cubed = (scaled_distance + epsilon)**3

    u = y - x - gravity_constant * \
        (x - origin_x) / scaled_distance_cubed - damping_factor * y
    v = -x - y - gravity_constant * \
        (y - origin_y) / scaled_distance_cubed + damping_factor * x

    return u, v
