#ifndef POLYNOMIAL_APPROXIMATE_HPP
#define POLYNOMIAL_APPROXIMATE_HPP

#include "Eigen/Core"
#include <random>

using namespace Eigen;

class PolynomialApproximate {
 public:
    PolynomialApproximate(int n) {
        x = VectorXd(n);
        for (int i = 0; i < n; i++) {
            x(i) = 1.0 * i / (n - 1);
        }
        y = (2 * M_PI * x).array().sin();

        /*
        // ノイズを乗せる場合
        std::mt19937 rand;
        std::normal_distribution<> dist(0.0, 0.05);
        
        for (int i = 0; i < n; i++) {
            y(i) += dist(rand);
        }
        */
    }

    VectorXd eval(const VectorXd &w) {
        return eval(w, x);
    }

    VectorXd eval(const VectorXd &w, const VectorXd &x) {
        VectorXd e = VectorXd::Zero(x.rows());
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

    // sinをn分割して誤差を求める
    double error(const VectorXd &w, int n) {
        VectorXd x = VectorXd(n);
        for (int i = 0; i < n; i++) {
            x(i) = 1.0 * i / (n - 1);
        }
        VectorXd y = (2 * M_PI * x).array().sin();
        return (y - eval(w, x)).array().pow(2).sum();
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

