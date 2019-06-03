#!/usr/bin/env python

import pickle
import numpy as np
from gptrials.robot import Robot

def evaluate_model(model, n_samples=1000):
    gpr = model
    r = Robot()

    def do_sample():
        q = np.random.uniform(0, 2*np.pi, 3)
        predicted = model.predict(q.reshape(1, -1))
        fkin = r.forward_kinematics(q)
        return (predicted - fkin).flatten()

    return np.array([do_sample() for _ in range(n_samples)])


def print_statistics(errors):
    for i, coor in zip([0, 1, 2], ['x', 'y', 'ϕ']):
        print(f"Average error in {coor}: {np.average(errors[:,i]):.6f}")
        print(f"Stddev on error in {coor}: {np.std(errors[:,i]):.6f}")

def plot_error_hists(errors):
    import matplotlib.pyplot as plt
    for i, coor in zip([0, 1, 2], ['x', 'y', 'ϕ']):
        plt.figure()
        plt.title(coor)
        plt.hist(errors[:,i], bins=100)
    plt.show()


if __name__ == "__main__":
    with open("trained_gp.pickle", "rb") as f:
        trained = pickle.load(f)

    errors = evaluate_model(trained)
    print_statistics(errors)
    plot_error_hists(errors)

