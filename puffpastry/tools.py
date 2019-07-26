import time
import numpy as np


class Timer(object):
    def __init__(self, msg=None):
        self.msg = msg

    def __enter__(self):
        self.tstart = time.time()
        return self

    def __exit__(self, type, value, traceback):
        dt = time.time() - self.tstart
        self.dt = dt
        if self.msg is not None:
            print(self.get_message())

    def get_elapsed_time(self):
        return self.dt

    def get_message(self):
        dt = self.dt
        prefix = 'Elapsed' if self.msg is None else self.msg
        if dt < 1:
            ss = '{}: {:.2f}ms'.format(prefix, dt*1e3)
        else:
            ss = '{}: {:.2f}s'.format(prefix, dt)
            if dt > 60:
                ss += ' ({:.2f}min)'.format(dt/60)
        return ss


def create_hypercube(dim, resolution):
    cube = np.array([np.linspace(0, 1, resolution)]).T
    for i in range(dim - 1):
        n = np.linspace(0, 1, resolution)
        tiled = np.tile(cube, (len(n), 1))
        repeated = np.repeat(n, cube.shape[0]).reshape((-1, 1))
        cube = np.hstack([repeated, tiled])
    return cube


