#ifndef POLYNOMIAL_APPROXIMATE_HPP
#define POLYNOMIAL_APPROXIMATE_HPP

#include "Eigen/Core"

using namespace Eigen;

class PolynomialApproximate {
 public:
    PolynomialApproximate(int n) {
        x = VectorXd(n);
        for (int i = 0; i < n; i++) {
            x(i) = 1.0 * i / (n - 1);
        }
        y = (2 * M_PI * x).array().sin();
    }

    VectorXd eval(const VectorXd &w) {
        VectorXd e = VectorXd::Zero(y.rows());
        for (int i = 0; i < w.rows(); i++) {
            e += VectorXd(w(i) * x.array().pow(i));
        }
        return e;
    }

    double operator()(const VectorXd &w) {
        return (y - eval(w)).array().pow(2).sum();
    }

    // 微分
    VectorXd diff(const VectorXd &w) {
        VectorXd res = VectorXd::Zero(w.rows());
        VectorXd d = y - eval(w);
        for (int i = 0; i < res.rows(); i++) {
            res(i) = -2.0 * (d.array() * x.array().pow(i)).sum();
        }
        return res;
    }

    double error(const VectorXd &w) {
        VectorXd y = (2 * M_PI * x).array().sin();
        return (y - eval(w)).array().pow(2).sum();
    }

    VectorXd getY() const {
        return y;
    }

    VectorXd getX() const {
        return x;
    }

 private:
    VectorXd y, x;
};

#endif

