from numpy import sin, cos
from functools import reduce
import numpy as np


class Robot(object):
    def __init__(self, link_lengths):
        self._link_lengths = link_lengths

    @staticmethod
    def __create_trafo(q, l):
        return np.array([[cos(q), -sin(q), l * cos(q)],
                         [sin(q), cos(q), l * sin(q)],
                         [0, 0, 1]])

    def forward_kinematics(self, joint_angles):
        trafos = [self.__create_trafo(q, l) for q, l in zip(joint_angles, self._link_lengths)]
        flange = reduce(lambda x, y: x.dot(y), trafos)
        return np.array([flange[0, 2], flange[1, 2], np.angle(np.exp(1j*(sum(joint_angles))))])
