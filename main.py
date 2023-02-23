import matplotlib.pyplot as plt
import numpy as np
import PIL

from vector import Vector2, vec2
from particle import Particle
from quadtree import Quadtree


def main():
    particles = [
        Particle(vec2(x, y), vec2(0,0), 1)
        for x in np.linspace(-5, 5, 4)
        for y in np.linspace(-5, 5, 4)
    ]

    tree = Quadtree(particles, vec2(0,0), 10)

    A = vec2(1, 2)
    B = vec2(7, 3)
    print(type(A-B))
    

if __name__ == "__main__":
    main()
