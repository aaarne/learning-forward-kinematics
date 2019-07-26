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


def sample_training_data(dim, n, robot):
    while True:
        q = np.random.uniform(0, 2*np.pi, (n, dim))
        forward_kin = np.apply_along_axis(robot.forward_kinematics, 1, q)
        smooth_fkin = np.zeros((forward_kin.shape[0], 4))
        smooth_fkin[:,0:2] = forward_kin[:,0:2]
        smooth_fkin[:,2] = np.cos(forward_kin[:,2])
        smooth_fkin[:,3] = np.sin(forward_kin[:,2])
        yield q, smooth_fkin


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
            Dense(16, input_shape=(self._dim,)),
            Activation('elu'),
            Dense(16),
            Activation('tanh'),
            Dense(4)
        ])

        nn.compile(optimizer='adam', loss=losses.mean_squared_error)

        with Timer("Model training"):
             history = nn.fit_generator(
                 sample_training_data(self._dim, 1024, robot),
                 validation_data=sample_training_data(self._dim, 128, robot),
                 validation_steps=1,
                 epochs=1000,
                 steps_per_epoch=8)
        import matplotlib.pyplot as plt
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.grid()
        plt.legend()

        return nn


def inference(model, joint_angles):
    raw = model.predict(joint_angles.reshape(1, -1)).flatten()
    return np.array([raw[0], raw[1], np.arctan2(raw[3], raw[2])])