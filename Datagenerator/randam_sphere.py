import math

import numpy as np
import matplotlib.pyplot as plt


def get_sphere_dataset(size: int = 100, radius: float = 2000.000, n_points: int = 2048, split: float = 0.85):
    dataset = np.zeros(shape=(size, n_points, 3))
    for i in range(size):
        sphere = sample_sphere_points(radius, n_points)
        dataset[i] = sphere

    split_index = int(round(size*split,0))

    train = dataset[:split_index]

    test = dataset[split_index:]

    return train, test


def sample_sphere_points(radius: int = 2000, samplesize: int = 2048):
    points = np.zeros(shape=(samplesize, 3))
    index = 0
    while index < samplesize:
        x = np.random.uniform(1.0, -1.0)
        y = np.random.uniform(1.0, -1.0)
        z = np.random.uniform(1.0, -1.0)

        if math.sqrt(x * x + y * y + z * z) < 1:
            points[index, 0] = x
            points[index, 1] = y
            points[index, 2] = z

            index += 1

    points = points * radius

    return points


if __name__ == '__main__':

    PLOT = False

    if PLOT is True:
        points = sample_sphere_points()

        fig = plt.figure(figsize=(2, 4))

        ax = fig.add_subplot(2, 4, 2, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=1)
        ax.set_axis_off()
        ax.title.set_text('Sphere')

        plt.show()

    train, test = get_sphere_dataset()

    print()


