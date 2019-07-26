#!/usr/bin/env python

import matplotlib.pyplot as plt
import pickle

from puffpastry.forward_kinematics_regression import inference
from puffpastry.implicit_function_viz import plot_implicit_function

with open("model.pickle", "rb") as f:
    m = pickle.load(f)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

plot_implicit_function(ax, lambda x: inference(m, x)[0], 0.0, color=(1, 0, 0, 0.2))
fig.tight_layout()
plt.show()
