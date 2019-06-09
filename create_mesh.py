#!/usr/bin/env python

import numpy as np
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


def create_mesh(func, level, interval, resolution):
    volume = evaluate_on_volume(func, interval, resolution)
    return marching_cubes(volume, level)


verts, faces, normal, values = create_mesh(lambda x: gp_inference(m, x)[0], 0.0, (0, 2*np.pi), 60)

filename = "manifold.obj"
import pymesh
mesh = pymesh.form_mesh(verts, faces)
pymesh.save_mesh(filename, mesh)
print(f"""Mesh stored as wavefront model to {filename}.
        You may want to solidify if using blender in order to print it.
        Select File -> Import -> Wavefront and then modifiers -> solidify.""")
