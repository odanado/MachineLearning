#include <iostream>
#include <random>
#include <array>
#include <functional>

#include "Eigen/Core"

using namespace std;
using namespace Eigen;

class DifferentialEvolution {
 public:
    using Func = function<double(const VectorXd &)>;

    DifferentialEvolution(const Func &func, double CR = 0.5,
                          double scaling = 0.6)
        : func(func), CR(CR), scaling(scaling) {}

    // 変数の数，解集団の数
    MatrixXd run(int iteration, int n, int m) {
        // 解集団
        auto population = randomMatrix(-1.0, 1.0, m, n);
        uniform_int_distribution<> distInt(0, population.cols() - 1);
        uniform_real_distribution<> dist(0, 1.0);

        while (iteration--) {
            for (int i = 0; i < population.rows(); i++) {
                const auto &x = population.row(i);
                auto pos = selectRandomly(i, population.rows());
                int j = distInt(rand);
                VectorXd new_x = VectorXd(population.cols());

                // Mutation
                VectorXd v =
                    population.row(pos[0]) +
                    scaling * (population.row(pos[1]) - population.row(pos[2]));

                // Crossover
                for (int k = 0; k < population.cols(); k++) {
                    if (k == 0 || CR < dist(rand)) {
                        new_x(j) = v(j);
                    } else {
                        new_x(j) = x(j);
                    }
                    j = (j + 1) % population.cols();
                }

                // Selection
                if (func(new_x) < func(x)) {
                    population.row(i) = new_x;
                }
            }
        }

        return population;
    }

    void setSeed(int seed) { rand.seed(seed); }

 private:
    array<int, 3> selectRandomly(int i, int n) {
        array<int, 3> pos;
        pos.fill(-1);
        int idx = 0;
        uniform_int_distribution<> dist(0, n - 1);

        while (idx < pos.size()) {
            int p = dist(rand);
            if (find(pos.begin(), pos.end(), p) == pos.end() && p != i)
                pos[idx++] = p;
        }
        return pos;
    }

    MatrixXd randomMatrix(double low, double high, int n, int m) {
        uniform_real_distribution<> dist(low, high);
        auto mat = MatrixXd(n, m);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                mat(i, j) = dist(rand);
            }
        }
        return mat;
    }

    Func func;
    double CR;
    double scaling;

    mt19937 rand;
};

class PolynomialApproximate {
 public:
    VectorXd y, x;
    PolynomialApproximate(int n) {
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

    double operator()(const VectorXd &w) {
        return (y - eval(w)).array().pow(2).sum();
    }
};

int main() {
    PolynomialApproximate pa(20);

    DifferentialEvolution de(pa);
    auto ws = de.run(10000, 9, 50);

    VectorXd w = ws.row(0);
    for (int i = 0; i < ws.rows(); i++) {
        if (pa(ws.row(i)) < pa(w)) w = ws.row(i);
    }
    fprintf(stderr, "error = %f\n", pa(w));

    for (int i = 0; i < pa.x.rows(); i++) {
        printf("%f %f %f\n", pa.x(i), pa.y(i), pa.eval(w)(i));
    }
}

