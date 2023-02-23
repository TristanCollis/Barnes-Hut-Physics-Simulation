from typing import Union
import numpy as np

from particle import Particle
from quadtree import Quadtree
from vector import Vector2, vec2

G = 1


def force_gravity(body1: Particle, body2: Particle | Quadtree) -> Vector2:
    r = body2.position - body1.position
    r_hat = r / r.norm

    return r_hat * G * body1.mass * body2.mass / r.norm_squared


def g_force_particle(particle: Particle, tree: Quadtree, theta=1) -> Vector2:
    if len(tree.particles) == 0:
        return vec2(0, 0)

    if len(tree.particles) == 1:
        return force_gravity(particle, tree.particles[0])

    if (tree.side_length / (particle.position - tree.position).norm) < theta:
        return force_gravity(particle, tree)

    return np.sum(
        g_force_particle(particle, child_tree, theta)
        for child_tree in tree.child_nodes)
