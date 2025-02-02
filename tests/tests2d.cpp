#include <random>
#include <tuple>

#include <gtest/gtest.h>

#include "convPolicies.h"
#include "usfft-seq.h"
#include "usfftpp.h"

using namespace usfftpp;

template <typename T>
void usdft2d_u2e(
    std::complex<T> *f,
    std::vector<std::tuple<T, T>> points,
    std::array<std::ptrdiff_t, 2> N,
    size_t M,
    int FourierType,
    std::complex<T> *a
) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < N[0]; i++) {
        for (int j = 0; j < N[1]; j++) {
            for (int k = 0; k < M; k++) {
                T ang = 2 * M_PI
                        * ((i - (int)N[0] / 2) * std::get<0>(points[k])
                           + (j - (int)N[1] / 2) * std::get<1>(points[k]))
                        * FourierType;
                a[i * N[1] + j] += f[k] * std::polar<T>(1, ang);
            }
        }
    }
}

template <typename T>
void usdft2d_e2u(
    std::complex<T> *f,
    std::vector<std::tuple<T, T>> points,
    std::array<std::ptrdiff_t, 2> N,
    size_t M,
    int FourierType,
    std::complex<T> *a
) {
#pragma omp parallel for
    for (int k = 0; k < M; k++) {
        for (int i = 0; i < N[0]; i++) {
            for (int j = 0; j < N[1]; j++) {
                T ang = 2 * M_PI
                        * ((j - (int)N[1] / 2) * std::get<1>(points[k])
                           + (i - (int)N[0] / 2) * std::get<0>(points[k]))
                        * FourierType;
                a[k] += f[i * N[1] + j] * std::polar<T>(1, ang);
            }
        }
    }
}

long double validate_one(
    std::complex<long double> *beta, std::complex<long double> *new_beta, int N
) {
    long double a, b;
    long double a_max = 0;
    long double b_max = 0;
    int i             = 0;
    std::complex<long double> c;
    for (i = 0; i < N; i++) {
        c = new_beta[i] - beta[i];
        a = std::abs(c);
        a_max += a * a;
        b = std::abs(new_beta[i]);
        b_max += b * b;
    }
    return a_max / b_max;
}

class USFFT2D_E2U : public ::testing::TestWithParam<std::tuple<int, int, int>> {
protected:
};

class USFFT2D_U2E : public ::testing::TestWithParam<std::tuple<int, int, int>> {
protected:
};

TEST_P(USFFT2D_E2U, Calculation_ShouldReturnCorrectResult) {
    using Plan2d = usfftpp::plan<
        float,
        2,
        usfftpp::seq<
            float,
            2,
            simple_par_visitor_policy<2>,
            simple_par_block_visitor_policy<2>>>;

    std::ptrdiff_t N1 = std::get<0>(GetParam());
    std::ptrdiff_t N2 = std::get<1>(GetParam());

    std::array<std::ptrdiff_t, 2> _N2d = { N1, N2 };

    std::ptrdiff_t points = std::get<2>(GetParam());

    std::vector<std::tuple<float, float>> v;
    float epsilon = 1e-4;

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-0.5, 0.5);

    for (ptrdiff_t j = 0; j < points; j++) {
        v.push_back({ distribution(generator), distribution(generator) });
    }

    Plan2d p(_N2d, v, epsilon);

    auto *F = new std::complex<float>[_N2d[0] * _N2d[1]];

    for (ptrdiff_t i = 0; i < _N2d[0] * _N2d[1]; i++) {
        F[i] = std::complex<float>(
            distribution(generator), distribution(generator)
        );
    }

    auto *f = new std::complex<float>[points];

    std::vector<std::tuple<long double, long double>> _x;
    for (ptrdiff_t i = 0; i < points; i++) {
        _x.push_back(std::make_tuple(std::get<0>(v[i]), std::get<1>(v[i])));
    }

    auto _F = new std::complex<long double>[_N2d[0] * _N2d[1]];
    for (ptrdiff_t i = 0; i < _N2d[0] * _N2d[1]; i++)
        _F[i] = F[i];

    auto *etalon = new std::complex<long double>[points];

    p.uninform_to_nonuniform_transform(F, f, fourier_direction::forward);

    usdft2d_e2u<long double>(_F, _x, _N2d, points, -1, etalon);

    delete[] _F;

    auto _f = new std::complex<long double>[points];
    for (int i = 0; i < points; i++)
        _f[i] = f[i];

    float err = validate_one(_f, etalon, points);

    EXPECT_LT(err, epsilon);

    delete[] etalon;
    delete[] f;
    delete[] _f;
}

TEST_P(USFFT2D_U2E, Calculation_ShouldReturnCorrectResult) {
    using Plan2d = usfftpp::plan<
        float,
        2,
        usfftpp::seq<
            float,
            2,
            simple_par_visitor_policy<2>,
            simple_par_block_visitor_policy<2>>>;

    std::ptrdiff_t N1 = std::get<0>(GetParam());
    std::ptrdiff_t N2 = std::get<1>(GetParam());

    std::array<std::ptrdiff_t, 2> _N2d = { N1, N2 };

    std::size_t points = std::get<2>(GetParam());

    std::vector<std::tuple<float, float>> v;
    float epsilon = 1e-4;

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-0.5, 0.5);

    for (int j = 0; j < points; j++) {
        v.push_back({ distribution(generator), distribution(generator) });
    }

    Plan2d *p = new Plan2d(_N2d, v, epsilon);

    auto *F = new std::complex<float>[points];

    for (int i = 0; i < points; i++) {
        F[i] = std::complex<float>(
            distribution(generator), distribution(generator)
        );
    }

    auto *f = new std::complex<float>[_N2d[0] * _N2d[1]];

    std::vector<std::tuple<long double, long double>> _x;
    for (int i = 0; i < points; i++) {
        _x.push_back(std::make_tuple(std::get<0>(v[i]), std::get<1>(v[i])));
    }

    auto _F = new std::complex<long double>[points];
    for (int i = 0; i < points; i++)
        _F[i] = F[i];

    auto *etalon = new std::complex<long double>[_N2d[0] * _N2d[1]];

    p->nonunform_to_uniform_transform(F, f, fourier_direction::forward);
    usdft2d_u2e<long double>(_F, _x, _N2d, points, -1, etalon);

    delete[] _F;

    auto _f = new std::complex<long double>[_N2d[0] * _N2d[1]];
    for (int i = 0; i < _N2d[0] * _N2d[1]; i++)
        _f[i] = f[i];

    float err = validate_one(_f, etalon, _N2d[0] * _N2d[1]);
    EXPECT_LT(err, epsilon);

    delete[] etalon;
    delete[] f;
    delete[] _f;
}

INSTANTIATE_TEST_SUITE_P(
    usfft2d,
    USFFT2D_E2U,
    ::testing::Combine(
        ::testing::Range(110, 114),
        ::testing::Range(110, 114),
        ::testing::Range(110, 114)
    )
);

INSTANTIATE_TEST_SUITE_P(
    usfft2d,
    USFFT2D_U2E,
    ::testing::Combine(
        ::testing::Range(110, 114),
        ::testing::Range(110, 114),
        ::testing::Range(110, 114)
    )
);
