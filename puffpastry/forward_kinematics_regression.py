from sklearn import gaussian_process as gp
from .tools import create_hypercube, Timer
import numpy as np


def _create_training_data(dim, resolution, robot):
    q = create_hypercube(dim, resolution)*2*np.pi
    forward_kin = np.apply_along_axis(robot.forward_kinematics, 1, q)
    smooth_fkin = np.zeros((forward_kin.shape[0], 4))
    smooth_fkin[:,0:2] = forward_kin[:,0:2]
    smooth_fkin[:,2] = np.cos(forward_kin[:,2])
    smooth_fkin[:,3] = np.sin(forward_kin[:,2])
    return q, smooth_fkin



class GPForwardKinematicsRegression(object):
    def __init__(self, dim=3, resolution=6):
        self._dim = dim
        self._resolution = resolution

    def fit(self, robot):
        g = gp.GaussianProcessRegressor()
        q, smooth_fkin = _create_training_data(self._dim, self._resolution, robot)

        print("Generated training set of {} samples.".format(q.shape[0]))
        with Timer("GP Regression"):
            g.fit(q, smooth_fkin)

        return g


class KerasForwardKinematicsRegression(object):
    def __init__(self, dim=3, resolution=6):
        self._dim = dim
        self._resolution = resolution

    def fit(self, robot):
        from keras.models import Sequential
        from keras.layers import Dense, Activation
        from keras import losses

        nn = Sequential([
            Dense(6, input_shape=(self._dim,)),
            Activation('sigmoid'),
            Dense(6),
            Activation('sigmoid'),
            Dense(4)
        ])
        q, smooth_fkin = _create_training_data(self._dim , self._resolution, robot)
        nn.compile(optimizer='adam', loss=losses.mean_squared_error)

        nn.fit(q, smooth_fkin, epochs=10, batch_size=32)





def inference(model, joint_angles):
    raw = model.predict(joint_angles.reshape(1, -1)).flatten()
    return np.array([raw[0], raw[1], np.arctan2(raw[3], raw[2])])