// 最急降下法
#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <chrono>


#include "Eigen/Core"

using namespace std;
using namespace Eigen;

class SteepestDescent {
 public:
    using DFunc = function<VectorXd(const VectorXd &)>;

    SteepestDescent(const DFunc &dfunc, double alpha = 0.01)
        : dfunc(dfunc), alpha(alpha) {}

    VectorXd run(int iteration, const VectorXd &initVec) {
        VectorXd x = initVec;
        while (iteration--) {
            VectorXd new_x = x - alpha * dfunc(x);
            x = new_x;
        }
        return x;
    }

    VectorXd run(const chrono::seconds &sec, const VectorXd &initVec) {
        VectorXd x = initVec;
        auto start = std::chrono::system_clock::now();

        while (std::chrono::system_clock::now() - start < sec) {
            VectorXd new_x = x - alpha * dfunc(x);
            x = new_x;
        }
        return x;
    }

 private:
    DFunc dfunc;
    double alpha;
};

class PolynomialApproximateDiff {
 public:
    VectorXd y, x;
    PolynomialApproximateDiff(int n) {
        x = VectorXd(n);
        for (int i = 0; i < n; i++) {
            x(i) = 1.0 * i / (n - 1);
        }
        y = 2 * M_PI * x;
        y = y.array().sin();
    }

    VectorXd eval(const VectorXd &w) {
        VectorXd e = VectorXd::Zero(y.rows());
        for (int i = 0; i < w.rows(); i++) {
            e += VectorXd(w(i) * x.array().pow(i));
        }
        return e;
    }

    VectorXd operator()(const VectorXd &w) {
        VectorXd res = VectorXd::Zero(w.rows());
        VectorXd d = y - eval(w);
        for (int i = 0; i < res.rows(); i++) {
            res(i) = -2.0 * (d.array() * x.array().pow(i)).sum();
        }
        return res;
    }

    double error(const VectorXd &w) {
        return (y - eval(w)).array().pow(2).sum();
    }
};

int main() {
    PolynomialApproximateDiff pad(20);
    SteepestDescent sd(pad);
    VectorXd w = VectorXd::Zero(9);
    w = sd.run(chrono::seconds(10), w);

    fprintf(stderr, "error = %f\n", pad.error(w));

    for (int i = 0; i < pad.x.rows(); i++) {
        printf("%f %f %f\n", pad.x(i), pad.y(i), pad.eval(w)(i));
    }

    return 0;
}
