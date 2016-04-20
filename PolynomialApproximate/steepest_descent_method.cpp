// 最急降下法
#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <chrono>

#include "Eigen/Core"
#include "polynomial_approximate.hpp"

using namespace std;
using namespace Eigen;

template<class Func>
class SteepestDescent {
 public:

    SteepestDescent(const Func &func, double alpha = 0.01)
        : func(func), alpha(alpha) {}

    VectorXd run(int iteration, const VectorXd &initVec) {
        VectorXd x = initVec;
        while (iteration--) {
            VectorXd new_x = x - alpha * func.diff(x);
            x = new_x;
        }
        return x;
    }

    VectorXd run(const chrono::seconds &sec, const VectorXd &initVec) {
        VectorXd x = initVec;
        auto start = std::chrono::system_clock::now();

        while (std::chrono::system_clock::now() - start < sec) {
            VectorXd new_x = x - alpha * func.diff(x);
            x = new_x;
        }
        return x;
    }

 private:
    Func func;
    double alpha;
};

int main(int argc,char *argv[]) {
    if (argc != 3) {
        return 1;
    }
    int pointCount = atoi(argv[1]);
    int degree = atoi(argv[2]);

    PolynomialApproximate pa(pointCount);
    SteepestDescent<PolynomialApproximate> sd(pa);
    VectorXd w = VectorXd::Zero(degree);
    w = sd.run(chrono::seconds(10), w);

    fprintf(stderr, "%d %d %f\n", pointCount, degree, pa.error(w, pointCount));

    auto y = pa.getY();
    auto x = pa.getX();

    for (int i = 0; i < x.rows(); i++) {
        printf("%f %f %f\n", x(i), y(i), pa.eval(w)(i));
    }

    return 0;
}

