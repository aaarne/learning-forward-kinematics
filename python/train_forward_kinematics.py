#!/usr/bin/env python3

import pickle

from gptrials.robot import Robot
from gptrials.reg import ForwardKinematicsRegression

if __name__ == "__main__":
    outfile = "trained_gp.pickle"
    r = Robot([1, 1, 1])

    fkr = ForwardKinematicsRegression(resolution=12)
    model = fkr.fit(r)
    with open(outfile, 'wb') as f:
        pickle.dump(model, f)
        print(f"Model written to {outfile}.")
