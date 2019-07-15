//
// Created by arne on 7/15/19.
//

#ifndef SHOW_MANIFOLD_ROBOT_H
#define SHOW_MANIFOLD_ROBOT_H


#include <Eigen/src/Core/Matrix.h>
#include <cmath>

using namespace Eigen;

class robot {

public:

    explicit robot(double l1, double l2, double l3) : l1(l1), l2(l2), l3(l3) {}

    Matrix3d fkin(double q1, double q2, double q3) {
        Matrix3d flange;
        flange = angle_to_trafo(l1, q1) * angle_to_trafo(l2, q2) * angle_to_trafo(l3, q3);
        return flange;
    }

    double fkin_y(double q1, double q2, double q3) {
        Matrix3d f = fkin(q1, q2, q3);
        return f(1, 2);
    }

    double fkin_omega(double q1, double q2, double q3) {
        Matrix3d f = fkin(q1, q2, q3);
        return atan2(f(1, 0), f(0, 0));
    }

private:
    Matrix3d angle_to_trafo(double l, double q) {
        Matrix3d m;
        m << cos(q), -sin(q), l * cos(q),
                sin(q), cos(q), l * sin(q),
                0, 0, 1;
        return m;
    }

    double l1, l2, l3;

};


#endif //SHOW_MANIFOLD_ROBOT_H
