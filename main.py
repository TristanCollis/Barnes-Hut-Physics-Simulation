import numpy as np
from PIL import Image as img

import forces
from procedures import simulation, rendering
from structs import Particle, RigidBody, Vector, rect, vec2

RNG = np.random.default_rng()


def rotation_matrix(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def noisy_particle(
    scale: float,
    mass: float,
    central_mass: float,
    angular_momentum: float,
) -> Particle:
    r = 0
    while not 0.5 * scale < r < 4 * scale:
        r = RNG.exponential(scale=scale)
    theta = RNG.random() * 2 * np.pi

    position = np.matmul(rotation_matrix(theta), vec2(r, 0))
    v_azimuthal = angular_momentum * scale / r * 1000
    v_radial = (
        ((angular_momentum / mass) ** 2 - 2 * forces.G * central_mass)
        * (1 / scale - 1 / r)
    ) ** 0.5

    v_radial *= RNG.choice((-1, 1))

    velocity = np.matmul(rotation_matrix(theta), vec2(v_radial, v_azimuthal))

    return Particle(position, velocity, mass)


def main():
    scale = 1000
    sim_dims = vec2(scale, scale) * 10
    window = rect.Rect(*rect.get_corners(vec2(0, 0), *sim_dims))
    resolution = [400] * 2

    N = 1000
    dt = 1
    num_frames = 100

    mass = 0.0001
    angular_momentum = N * mass**0.67 * .1  # Per particle

    forces.G = scale * N * mass / 1000

    particles = [
        noisy_particle(scale, mass, N * mass, angular_momentum) for _ in range(N)
    ]

    points = [particle.position for particle in particles]

    velocities = [particle.velocity for particle in particles]

    masses = [mass] * N

    position_history = simulation.verlet_quadtree(
        points, velocities, masses, simulation.tree_gravity, dt, num_frames, window, 1
    )

    animation = rendering.animation_from_positions(position_history, window, resolution)  # type: ignore

    rendering.save_animation(animation, "out.gif")


if __name__ == "__main__":
    main()
