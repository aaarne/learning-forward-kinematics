#!/usr/bin/env python

import numpy as np
import pickle

from puffpastry.implicit_function_viz import create_mesh
from puffpastry.forward_kinematics_regression import inference

with open("model.pickle", "rb") as f:
    m = pickle.load(f)

print("Loaded model.")

print("Start evaluation")
mesh = create_mesh(lambda x: inference(m, x)[0], 0.0, (0, 2 * np.pi), 120)
print("Finished evaluation")

filename = "manifold.obj"
import pymesh
pymesh.save_mesh(filename, mesh)
print(f"""Mesh stored as wavefront model to {filename}.
        You may want to solidify if using blender in order to print it.
        Select File -> Import -> Wavefront and then modifiers -> solidify.""")
