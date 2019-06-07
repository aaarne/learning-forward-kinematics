#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes_lewiner as marching_cubes
import pickle

with open("model.pickle", "rb") as f:
    m = pickle.load(f)


def gp_inference(model, joint_angles):
    raw = model.predict(joint_angles.reshape(1, -1)).flatten()
    return np.array([raw[0], raw[1], np.arctan2(raw[3], raw[2])])


def evaluate_on_volume(func, interval, resolution):
    data = np.zeros((resolution, resolution, resolution))
    ax = np.linspace(interval[0], interval[1], resolution)
    for i1, x1 in enumerate(ax):
        for i2, x2 in enumerate(ax):
            for i3, x3 in enumerate(ax):
                data[i1, i2, i3] = func(np.array([x1, x2, x3]))
    return data


def plot_implicit_function(ax, func, level,
                           interval=(0, 2 * np.pi),
                           resolution=24,
                           color=((0.5, 0.5, 0.5, 0.5))):
    volume = evaluate_on_volume(func, interval, resolution)
    verts, faces, normals, values = marching_cubes(volume, level)
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor((0, 0, 0, 0))
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)
    for i in [ax.set_xlim, ax.set_ylim, ax.set_zlim]:
        i(0, resolution)
    for g, s in zip([ax.get_xticks, ax.get_yticks, ax.get_zticks],
                    [ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels]):
        s([f"{v:.3f}" for v in (g() * (interval[1] - interval[0]) / resolution + interval[0])])


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

plot_implicit_function(ax, lambda x: gp_inference(m, x)[0], 0.0, color=(1, 0, 0, 0.2))
plot_implicit_function(ax, lambda x: gp_inference(m, x)[0], 1.0, color=(0, 1, 0, 0.2))
plot_implicit_function(ax, lambda x: gp_inference(m, x)[0], 2.0, color=(1, 0.5, 0, 0.2))
plot_implicit_function(ax, lambda x: gp_inference(m, x)[0], 3.0, color=(0, 0, 1, 0.2))
fig.tight_layout()
plt.show()
