from typing import Any, Callable, Sequence

import numpy as np

import forces
from structs import Particle, Quadtree, Rect, RigidBody, Vector, norm, vec2, in_bounds, cull


def integrate_euler(
    particles: Sequence[Particle], forces: Sequence[Vector], dt: float
) -> None:
    for particle, force in zip(particles, forces):
        particle.velocity += force * dt / particle.mass
        particle.position += particle.velocity * dt


def tree_gravity(
    point: Vector, mass: float, tree: Quadtree, theta: float = 1
) -> Vector:
    if norm(point - tree.position) / tree.diameter <= theta:
        return forces.gravity(point, tree.position, mass, tree.mass)

    if len(tree.points) == 0:
        return vec2(0, 0)

    if len(tree.points) == 1:
        return forces.gravity(point, tree.points[0], mass, tree.mass)

    return sum(
        tree_gravity(point, mass, subtree, theta) for subtree in tree.child_nodes
    )  # type: ignore


def brute_force(
    particles: Sequence[Particle],
    force_func: Callable[[Particle, Particle], Vector],
    dt: float,
    num_frames: int,
) -> np.ndarray[float, Any]:
    position_history = np.zeros((num_frames, len(particles), 2))

    for i in range(num_frames):
        print(f"{i} / {num_frames}")

        forces = [
            force_func(particle, other)
            for particle in particles
            for other in particles
            if other is not particle
        ]

        integrate_euler(particles, forces, dt)

        position_history[i] = np.array([particle.position for particle in particles])

    return position_history


def verlet_quadtree(
    points: Sequence[Vector],
    velocities: Sequence[Vector],
    masses: Sequence[float],
    force_func: Callable[[Vector, float, Quadtree, float], Vector],
    dt: float,
    num_frames: int,
    bounds: Rect,
    theta: float = 1,
) -> np.ndarray[float, Any]:
    
    point_history = np.zeros((num_frames, len(points), 2))
    point_history[0] = points

    velocity_history = np.zeros_like(point_history)
    velocity_history[0] = velocities

    acceleration_history = np.zeros_like(point_history)

    _masses = np.array(masses, copy=True)

    for j in range(len(points)):
        if not in_bounds(point_history[0, j], bounds):
            point_history[0, j] = vec2(0, 0)
            velocity_history[0, j] = vec2(0, 0)
            _masses[j] = 0

    for i in range(num_frames):
        # print(f"{i} / {num_frames}")

        tree = Quadtree(point_history[i], bounds, _masses)  # type: ignore
        for j, (point, mass) in enumerate(zip(point_history[i], _masses)):
            if mass == 0:
                continue

            acceleration_history[i, j] = force_func(point, mass, tree, theta) / mass

        if i == 0:
            continue 

        point_history[i] = (
            point_history[i - 1]
            + velocity_history[i - 1] * dt
            + 0.5 * acceleration_history[i - 1] * dt**2
        )

        for j in range(len(points)):
            if not in_bounds(point_history[i, j], bounds):
                point_history[i, j] = vec2(0, 0)
                velocity_history[i-1, j] = vec2(0, 0)
                _masses[j] = 0

        velocity_history[i] = velocity_history[i-1] + (acceleration_history[i] + acceleration_history[i-1]) / 2 * dt

    return point_history
