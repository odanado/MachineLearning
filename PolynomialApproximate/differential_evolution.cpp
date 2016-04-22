#include <iostream>
#include <random>
#include <array>
#include <functional>
#include <chrono>
#include <vector>
#include <atomic>

#include "Eigen/Core"
#include "polynomial_approximate.hpp"

using namespace std;
using namespace Eigen;

template<class Func>
class DifferentialEvolution {
 public:
     int trialCount;

    DifferentialEvolution(const Func &func, double CR = 0.5,
                          double scaling = 0.6)
        : func(func), CR(CR), scaling(scaling), trialCount(0) {}

    // 変数の数，解集団の数
    MatrixXd run(int iteration, int n, int m) {
        init(n, m);

        while (iteration--) {
            trial(population, distInt, dist);
        }

        return population;
    }

    // 変数の数，解集団の数
    MatrixXd run(const chrono::seconds &sec, int n, int m) {
        init(n, m);
        auto start = std::chrono::system_clock::now();

        while (std::chrono::system_clock::now() - start < sec) {
            trial(population, distInt, dist);
        }

        cout << trialCount << endl;
        return population;
    }

    void setSeed(int seed) { rand.seed(seed); }

 private:
    void trial(MatrixXd &population, uniform_int_distribution<> &distInt,
               uniform_real_distribution<> &dist) {

        trialCount++;
        MatrixXd prevPopulation = population;
        for (int i = 0; i < population.rows(); i++) {
            const auto &x = prevPopulation.row(i);
            auto pos = selectRandomly(i, population.rows());
            int j = distInt(rand);
            VectorXd new_x = VectorXd(population.cols());

            // Mutation
            VectorXd v =
                prevPopulation.row(pos[0]) +
                scaling * (prevPopulation.row(pos[1]) - prevPopulation.row(pos[2]));

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
            auto newScore = func(new_x);
            if(newScore < bestScore[i]) {
                population.row(i) = new_x;
                bestScore[i] = newScore;
            }
        }
    }

    void init(int n, int m) {
        population = randomMatrix(-1.0, 1.0, m, n);
        distInt = uniform_int_distribution<>(0, population.cols() - 1);
        dist = uniform_real_distribution<>(0, 1.0);
        for (int i = 0; i < m; i++) {
           bestScore.push_back(func(population.row(i)));
        }
    }

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

    // 解集団
    MatrixXd population;
    // 一様乱数生成
    uniform_int_distribution<> distInt;
    uniform_real_distribution<> dist;

    // 各解ごとに一番良いものを覚えておく
    vector<double> bestScore;
};

int main(int argc,char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "点の数 次数\n");
        return 1;
    }
    int pointCount = atoi(argv[1]);
    int degree = atoi(argv[2]) + 1;

    PolynomialApproximate pa(pointCount);

    DifferentialEvolution<PolynomialApproximate> de(pa);
    auto ws = de.run(chrono::seconds(10), degree, 50);

    VectorXd w = ws.row(0);
    for (int i = 0; i < ws.rows(); i++) {
        if (pa(ws.row(i)) < pa(w)) w = ws.row(i);
    }
    fprintf(stderr, "%d %d %f\n", pointCount, degree, pa.error(w, pointCount));

    auto y = pa.getY();
    auto x = pa.getX();

    for (int i = 0; i < x.rows(); i++) {
        printf("%f %f %f\n", x(i), y(i), pa.eval(w)(i));
    }
}

