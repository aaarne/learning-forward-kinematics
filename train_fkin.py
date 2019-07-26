#!/usr/bin/env python

import numpy as np

from puffpastry.planar_robot import PlanarRobot
from puffpastry.forward_kinematics_regression import *
from puffpastry.tools import Timer


test_robot = PlanarRobot([1, 1, 1])
resolution = 60

#model = GPForwardKinematicsRegression(resolution=12).fit(test_robot)
model = KerasForwardKinematicsRegression(resolution=resolution).fit(test_robot)

def evaluate_model(model, n_samples=10000):
    gpr = model

    def do_sample():
        q = np.random.uniform(0, 2*np.pi, 3)
        predicted = inference(model, q)
        fkin = test_robot.forward_kinematics(q)
        return (predicted - fkin).flatten()

    with Timer() as t:
        e = np.array([do_sample() for _ in range(n_samples)])

    dt = 1e3*t.get_elapsed_time()/n_samples
    print("Inferene in model takes {:.3f}ms per sample.".format(dt))
    return e


result = evaluate_model(model)

def print_statistics(e):
    for i, coor in zip([0, 1, 2], ['x', 'y', 'ϕ']):
        d = np.abs(e[:, i])
        print("{}-direction:\tmax: {:.6f}\taverage: {:.6f}\tstddev: {:.6f}".format(coor, np.max(d), np.average(d), np.std(d)))


print_statistics(result)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3)

for index, coor in enumerate(['x', 'y', 'φ']):
    ax = axes[index]
    ax.hist(result[:,index], bins=100)
    ax.set_title("Error in {}-direction".format(coor))
    ax.grid()
    if index in [0, 1]:
        ax.set_xlabel("Deviation in {}-direction in [m]")
    else:
        ax.set_xlabel("Orientation deviation in rad")

plt.show()

import pickle
with open("model.pickle", "wb") as f:
    pickle.dump(model, f)
