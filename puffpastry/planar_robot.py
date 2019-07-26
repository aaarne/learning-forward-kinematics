import numpy as np
from numpy import sin, cos
from functools import reduce


class PlanarRobot(object):
    def __init__(self, link_lengths=[1, 1, 1]):
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

    def forward_kinematics_deg(self, joint_angles_deg):
        return self.forward_kinematics(np.deg2rad(joint_angles_deg))