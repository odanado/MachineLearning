// 最急降下法
#include <iostream>
#include <random>
#include <cmath>
#include <vector>

using Vector = std::vector<double>;

double func(const Vector &x) {
    return x[0] * x[0] - 2 * x[0] + 4 * x[1] * x[1];
}

/*
Vector dfunc(const Vector &x) {
    Vector v(x.size());
    v[0] = 2 * x[0] - 2;
    v[1] = 8 * x[1];
    return v;
}
*/

Vector add(const Vector &v1, const Vector &v2) {
    Vector res(v1.size());
    for (int i = 0; i < res.size(); i++) {
        res[i] = v1[i] + v2[i];
    }
    return res;
}

Vector minus(const Vector &v) {
    Vector res(v.size());
    for (int i = 0; i < res.size(); i++) {
        res[i] = -v[i];
    }
    return res;
}

Vector mul(double d, const Vector &v) {
    Vector res(v.size());
    for (int i = 0; i < res.size(); i++) {
        res[i] = d * v[i];
    }
    return res;
}

double norm(const Vector &v) {
    double res = 0.0;
    for (auto &&var : v) {
        res += var * var;
    }
    return res;
}

void print(const Vector &v) {
    for (auto &&var : v) {
        std::cout << var << " ";
    }
    std::cout << std::endl;
}

struct Dfunc {
    Vector x, y;
    Dfunc(int n) {
        for (int i = 0; i <= 20; i++) {
            x.push_back(2.0 * i / n);
            y.push_back(sin(M_PI * x.back()));
        }
    }

    Vector operator()(const Vector &w) {
        Vector v(w.size());
        for (int i = 0; i < v.size(); i++) {
            v[i] = df(i, w);
        }
        return v;
    }

    double df(int k, const Vector &w) {
        double d = 0.0;
        for (int i = 0; i < x.size(); i++) {
            d += diff(i, w) * pow(x[i], k);
        }
        return -2.0 * d;
    }

    double eval(double x, const Vector &w) {
        double e = 0.0;
        for (int i = 0; i < w.size(); i++) {
            e += w[i] * pow(x, i);
        }
        return e;
    }

    double diff(int i, const Vector &w) { return y[i] - eval(x[i], w); }
    void check(const Vector &w) {
        for (int i = 0; i < y.size(); i++) {
            std::cout << y[i] << ": " << eval(x[i], w) << std::endl;
        }
    }
    void print_py(const Vector &w) {
        for (int i = 0; i < x.size(); i++) {
            std::cout << eval(x[i], w) << ", ";
        }
        std::cout<<std::endl;
    }
};

void print_py(const Vector &v) {
    for (auto &&var : v) {
        std::cout << var << ", ";
    }
    std::cout << std::endl;
}

int main() {
    double alpha = 0.01;
    Vector x(20, 0);
    Dfunc dfunc(x.size());

    int n;
    std::cin>>n;
    for (int i = 0; i < n; i++) {
        auto d = minus(dfunc(x));
        auto new_x = add(x, mul(alpha, d));
        // print(new_x);
        //std::cout << norm(d) << std::endl;
        x = new_x;
    }
    print_py(dfunc.x);
    print_py(dfunc.y);
    print_py(x);
    dfunc.check(x);
    dfunc.print_py(x);
    return 0;
}
