from sklearn import gaussian_process as gp
import numpy as np


class ForwardKinematicsRegression(object):
    def __init__(self, dim=3, resolution=360):
        self._dim = dim
        self._resolution = resolution

    def fit(self, robot):
        g = gp.GaussianProcessRegressor()

        q = np.array([np.linspace(0, 360, self._resolution)]).T
        for i in range(self._dim - 1):
            qnew = np.linspace(0, 360, self._resolution)
            tiled = np.tile(q, (len(qnew), 1))
            repeated = np.repeat(qnew, q.shape[0]).reshape((-1, 1))
            q = np.hstack([repeated, tiled])

        q = np.deg2rad(q)
        forward_kin = np.apply_along_axis(robot.forward_kinematics, 1, q)
        print(f"Generated training set of {q.shape[0]} samples.")
        print("Start training...")
        g.fit(q, forward_kin)
        print("Finished training.")
        return g
