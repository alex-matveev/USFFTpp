#include <random>
#include <tuple>

#include <gtest/gtest.h>

#include "convPolicies.h"
#include "usfft-seq.h"
#include "usfftpp.h"

using namespace usfftpp;

template <typename T>
void usdft1d_e2u(
    std::complex<T> *f,
    std::vector<T> w,
    std::array<std::ptrdiff_t, 1> N,
    std::ptrdiff_t M,
    int FourierType,
    std::complex<T> *a
) {
#pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < M; i++) {
        for (std::ptrdiff_t j = 0; j < N[0]; j++) {
            T ang = 2 * M_PI * (j - (std::ptrdiff_t)N[0] / 2) * w[i] * FourierType;
            a[i] += f[j] * std::polar<T>(1, ang);
        }
    }
}

template <typename T>
void usdft1d_u2e(
    std::complex<T> *f,
    std::vector<T> w,
    std::array<std::ptrdiff_t, 1> N,
    std::ptrdiff_t M,
    int FourierType,
    std::complex<T> *a
) {
#pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < N[0]; i++) {
        for (std::ptrdiff_t j = 0; j < M; j++) {
            T ang = 2 * M_PI * (i - (std::ptrdiff_t)N[0] / 2) * w[j] * FourierType;
            a[i] += f[j] * std::polar<T>(1, ang);
        }
    }
}

long double validate_one(
    std::vector<long double> x,
    std::complex<long double> *beta,
    std::complex<long double> *new_beta,
    int N
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

class USFFT1D_E2U : public ::testing::TestWithParam<std::tuple<int, int>> {
protected:
};

class USFFT1D_U2E : public ::testing::TestWithParam<std::tuple<int, int>> {
protected:
};

TEST_P(USFFT1D_E2U, Calculation_ShouldReturnCorrectResult) {
    using Plan1d = usfftpp::plan<
        float,
        1,
        usfftpp::seq<
            float,
            1,
            simple_par_visitor_policy<1>,
            simple_par_block_visitor_policy<1>>>;

    std::ptrdiff_t N                   = std::get<0>(GetParam());
    std::array<std::ptrdiff_t, 1> _N1d = { N };

    std::ptrdiff_t points = std::get<1>(GetParam());

    std::vector<float> vx;
    float epsilon = 1e-4;

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-0.5, 0.5);

    for (std::ptrdiff_t j = 0; j < points; j++) {
        vx.push_back(0.0);
    }

    Plan1d p(_N1d, vx, epsilon);

    auto *F = new std::complex<float>[_N1d[0]];

    for (std::ptrdiff_t i = 0; i < _N1d[0]; i++) {
        F[i] = { 5, 0 };
    }

    auto *f = new std::complex<float>[points];

    std::vector<long double> _x;
    for (std::ptrdiff_t i = 0; i < points; i++) {
        _x.push_back(vx[i]);
    }

    auto _F = new std::complex<long double>[_N1d[0]];
    for (std::ptrdiff_t i = 0; i < _N1d[0]; i++)
        _F[i] = F[i];

    auto *etalon = new std::complex<long double>[points];

    usdft1d_e2u<long double>(_F, _x, _N1d, points, -1, etalon);

    p.uninform_to_nonuniform_transform(F, f, fourier_direction::forward);
    delete[] _F;

    auto _f = new std::complex<long double>[points];
    for (std::ptrdiff_t i = 0; i < points; i++)
        _f[i] = f[i];

    float err = validate_one(_x, _f, etalon, points);

    EXPECT_LT(err, epsilon);

    delete[] etalon;
    delete[] f;
    delete[] _f;
}

TEST_P(USFFT1D_U2E, Calculation_ShouldReturnCorrectResult) {
    using Plan1d = plan<
        float,
        1,
        seq<float,
            1,
            simple_par_visitor_policy<1>,
            simple_par_block_visitor_policy<1>>>;
    std::ptrdiff_t N                   = std::get<0>(GetParam());
    std::array<std::ptrdiff_t, 1> _N1d = { N };

    std::ptrdiff_t points = std::get<1>(GetParam());
    
    std::vector<float> vx;
    float epsilon = 1e-6;

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-0.5, 0.5);
    std::uniform_real_distribution<float> distribution2(-0.5, 0.5);

    for (std::ptrdiff_t j = 0; j < points; j++) {
        vx.push_back(0.0);
    }

    Plan1d p(_N1d, vx, epsilon);

    auto *F = new std::complex<float>[points];

    for (std::ptrdiff_t i = 0; i < points; i++) {
        F[i] = std::complex<float>(
            distribution2(generator), distribution2(generator)
        );
    }

    auto *f = new std::complex<float>[_N1d[0]];

    std::vector<long double> _x;
    for (std::ptrdiff_t i = 0; i < points; i++) {
        _x.push_back(vx[i]);
    }

    auto _F = new std::complex<long double>[points];
    for (std::ptrdiff_t i = 0; i < points; i++)
        _F[i] = F[i];

    auto etalon = new std::complex<long double>[_N1d[0]];

    usdft1d_u2e<long double>(_F, _x, _N1d, points, -1, etalon);

    p.nonunform_to_uniform_transform(F, f, fourier_direction::forward);
    delete[] _F;

    auto _f = new std::complex<long double>[_N1d[0]];
    for (std::ptrdiff_t i = 0; i < _N1d[0]; i++)
        _f[i] = f[i];

    float err = validate_one(_x, _f, etalon, _N1d[0]);
    EXPECT_LT(err, epsilon);

    delete[] etalon;
    delete[] f;
    delete[] _f;
}

INSTANTIATE_TEST_SUITE_P(
    usfft1d,
    USFFT1D_E2U,
    ::testing::Combine(
        ::testing::Range(97, 108), ::testing::Range(97, 108)
    )
);
INSTANTIATE_TEST_SUITE_P(
    usfft1d,
    USFFT1D_U2E,
    ::testing::Combine(::testing::Range(97, 108), ::testing::Range(97, 108))
);
