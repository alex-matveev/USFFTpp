#include <iostream>

#include "usfft-seq.h"
#pragma once

namespace usfftpp {

template <typename T, typename GatherVisitorPolicy, typename ScatterVisitorPolicy>
void seq<T, 1, GatherVisitorPolicy, ScatterVisitorPolicy>::gather(std::complex<T> *out,
                                                                  std::complex<T> *buffer) {
    this->fill_borders(buffer);

    auto exp_square = std::make_unique<T[]>(2 * this->m_radius + 1);

    for (std::ptrdiff_t i = 0; i < 2 * this->m_radius + 1; i++) {
        exp_square[i] = exp(-i * i * this->m_lambda);
    }
    auto lambda = [this, exp_square = exp_square.get()](std::ptrdiff_t i, std::complex<T> *f,
                                                        std::complex<T> *buffer) {
        std::complex<T> res = 0;

        auto mux_j = static_cast<std::ptrdiff_t>(2 * this->m_points[i] * this->m_N[0]);
        T dx = fabs(mux_j - this->m_radius -
                    this->m_points[i] * (this->m_oversamplingFactor * this->m_N[0]));
        T weight_acc_x = 1;
        T weight_zero_x = exp(-this->m_lambda * dx * dx);
        T weight_step_x = exp(2 * this->m_lambda * dx);

        std::ptrdiff_t ind = this->m_strides[0] +
                             this->m_strides[1] * (mux_j - this->m_radius +
                                                   this->m_oversamplingFactor * this->m_N[0] / 2);
        for (std::ptrdiff_t r = 0; r < 2 * this->m_radius + 1; r++) {
            res += weight_zero_x * weight_acc_x * buffer[ind + r] * exp_square[r];
            weight_acc_x *= weight_step_x;
        }
        f[this->m_di[i].second] = res;
    };

    GatherVisitorPolicy::run(lambda, this->m_points, out, buffer);
}

template <typename T, typename GatherVisitorPolicy, typename ScatterVisitorPolicy>
void seq<T, 2, GatherVisitorPolicy, ScatterVisitorPolicy>::gather(std::complex<T> *out,
                                                                  std::complex<T> *buffer) {
    this->fill_borders(buffer);

    auto exp_square = std::make_unique<T[]>(2 * this->m_radius + 1);

    for (std::ptrdiff_t i = 0; i < 2 * this->m_radius + 1; i++) {
        exp_square[i] = exp(-i * i * this->m_lambda);
    }
    auto lambda = [this, exp_square = exp_square.get()](std::ptrdiff_t i, std::complex<T> *out,
                                                        std::complex<T> *buffer, T *local_weight_x,
                                                        T *local_weight_y) {
        std::complex<T> res = 0;

        auto mux_j = static_cast<std::ptrdiff_t>(this->m_oversamplingFactor *
                                                 std::get<1>(this->m_points[i]) * this->m_N[1]);
        auto muy_j = static_cast<std::ptrdiff_t>(this->m_oversamplingFactor *
                                                 std::get<0>(this->m_points[i]) * this->m_N[0]);

        T dx = fabs(mux_j - this->m_radius -
                    std::get<1>(this->m_points[i]) * (this->m_oversamplingFactor * this->m_N[1]));
        T dy = fabs(muy_j - this->m_radius -
                    std::get<0>(this->m_points[i]) * (this->m_oversamplingFactor * this->m_N[0]));

        T weight_acc_x = 1;
        T weight_zero_x = exp(-this->m_lambda * dx * dx);
        T weight_step_x = exp(2 * this->m_lambda * dx);

        T weight_acc_y = 1;
        T weight_zero_y = exp(-this->m_lambda * dy * dy);
        T weight_step_y = exp(2 * this->m_lambda * dy);

        local_weight_x[0] = weight_zero_x;
        local_weight_y[0] = weight_zero_y;

        for (std::ptrdiff_t k = 0; k < 2 * this->m_radius + 1; k++) {
            local_weight_x[k] = weight_zero_x * weight_acc_x * exp_square[k];
            local_weight_y[k] = weight_zero_y * weight_acc_y * exp_square[k];
            weight_acc_x *= weight_step_x;
            weight_acc_y *= weight_step_y;
        }

        for (std::ptrdiff_t k = -this->m_radius; k <= this->m_radius; k++) {
            for (std::ptrdiff_t l = -this->m_radius; l <= this->m_radius; l++) {
                res += local_weight_x[l + this->m_radius] * local_weight_y[k + this->m_radius] *
                       buffer[this->m_strides[0] +
                              this->m_strides[2] *
                                  (std::ptrdiff_t)(mux_j + l +
                                                   this->m_oversamplingFactor * this->m_N[1] / 2) +
                              this->m_strides[1] *
                                  (std::ptrdiff_t)(muy_j + k +
                                                   this->m_oversamplingFactor * this->m_N[0] / 2)];
            }
        }

        out[this->m_di[i].second] = res;
    };

    GatherVisitorPolicy::run(lambda, this->m_points, out, buffer,
                             2 * (2 * this->m_radius + 1) * sizeof(T));
}

template <typename T, typename GatherVisitorPolicy, typename ScatterVisitorPolicy>
void seq<T, 1, GatherVisitorPolicy, ScatterVisitorPolicy>::scatter(std::complex<T> *in,
                                                                   std::complex<T> *buffer) {

    auto exp_square = std::make_unique<T[]>(2 * this->m_radius + 1);

    for (std::ptrdiff_t i = 0; i < 2 * this->m_radius + 1; i++) {
        exp_square[i] = exp(-i * i * this->m_lambda);
    }
    this->fill_first_occurrence_array();
    auto tmp_in = std::make_unique<std::complex<T>[]>(this->m_points.size());
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < this->m_points.size(); i++) {
        tmp_in[i] = in[this->m_di[i].second];
    }

    auto lambda = [this, exp_square = exp_square.get()](std::ptrdiff_t i, std::complex<T> *in,
                                                        std::complex<T> *buffer) {
        auto vx = static_cast<std::ptrdiff_t>(2 * this->m_N[0] * this->m_points[i]);

        std::ptrdiff_t ind =
            this->m_strides[0] + this->m_strides[1] * (vx - this->m_radius + 2 * this->m_N[0] / 2);
        T dx = fabs(vx - this->m_radius - this->m_points[i] * static_cast<T>(2 * this->m_N[0]));
        T weight_acc_x = 1;
        T weight_zero_x = exp(-this->m_lambda * dx * dx);
        T weight_step_x = exp(2 * this->m_lambda * dx);
        for (std::ptrdiff_t r = 0; r < 2 * this->m_radius + 1; r++) {
            buffer[ind + r] += weight_zero_x * weight_acc_x * exp_square[r] * in[i];
            weight_acc_x *= weight_step_x;
        }
    };

    ScatterVisitorPolicy::run(lambda, this->m_points, tmp_in.get(), this->m_N,
                              this->m_firstOccurrenceArray.get(), buffer);
    this->wrap_borders(buffer);
}

template <typename T, typename GatherVisitorPolicy, typename ScatterVisitorPolicy>
void seq<T, 2, GatherVisitorPolicy, ScatterVisitorPolicy>::scatter(std::complex<T> *in,
                                                                   std::complex<T> *buffer) {
    auto exp_square = std::make_unique<T[]>(2 * this->m_radius + 1);

    for (std::ptrdiff_t i = 0; i < 2 * this->m_radius + 1; i++) {
        exp_square[i] = exp(-i * i * this->m_lambda);
    }

    this->fill_first_occurrence_array();

    auto tmp_in = std::make_unique<std::complex<T>[]>(this->m_points.size());

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < this->m_points.size(); i++) {
        tmp_in[i] = in[this->m_di[i].second];
    }

    auto lambda = [this, exp_square = exp_square.get()](std::size_t i, std::complex<T> *in,
                                                        std::complex<T> *buffer, T *local_weight_x,
                                                        T *local_weight_y) {
        auto vx_j = static_cast<std::ptrdiff_t>(this->m_oversamplingFactor * this->m_N[1] *
                                                std::get<1>(this->m_points[i]));
        auto vy_j = static_cast<std::ptrdiff_t>(this->m_oversamplingFactor * this->m_N[0] *
                                                std::get<0>(this->m_points[i]));

        T dx = fabs(vx_j - this->m_radius -
                    std::get<1>(this->m_points[i]) * (this->m_oversamplingFactor * this->m_N[1]));
        T dy = fabs(vy_j - this->m_radius -
                    std::get<0>(this->m_points[i]) * (this->m_oversamplingFactor * this->m_N[0]));

        T weight_acc_x = 1;
        T weight_zero_x = exp(-this->m_lambda * dx * dx);
        T weight_step_x = exp(2 * this->m_lambda * dx);

        T weight_acc_y = 1;
        T weight_zero_y = exp(-this->m_lambda * dy * dy);
        T weight_step_y = exp(2 * this->m_lambda * dy);

        local_weight_x[0] = weight_zero_x;
        local_weight_y[0] = weight_zero_y;

        for (std::ptrdiff_t r = 0; r < 2 * this->m_radius + 1; r++) {
            local_weight_x[r] = weight_zero_x * weight_acc_x * exp_square[r];
            local_weight_y[r] = weight_zero_y * weight_acc_y * exp_square[r];
            weight_acc_x *= weight_step_x;
            weight_acc_y *= weight_step_y;
        }

        for (std::ptrdiff_t ry = -this->m_radius; ry <= this->m_radius; ry++) {
            for (std::ptrdiff_t rx = -this->m_radius; rx <= this->m_radius; rx++) {
                buffer[this->m_strides[0] +
                       this->m_strides[2] *
                           (std::ptrdiff_t)(vx_j + rx +
                                            this->m_oversamplingFactor * this->m_N[1] / 2) +
                       this->m_strides[1] *
                           (std::ptrdiff_t)(vy_j + ry +
                                            this->m_oversamplingFactor * this->m_N[0] / 2)] +=
                    local_weight_x[rx + this->m_radius] * local_weight_y[ry + this->m_radius] *
                    in[i];
            }
        }
    };

    ScatterVisitorPolicy::run(lambda, this->m_points, tmp_in.get(), this->m_N,
                              this->m_firstOccurrenceArray.get(), buffer,
                              this->m_oversamplingFactor, 2 * this->m_radius + 1,
                              2 * (2 * this->m_radius + 1) * sizeof(T));
    this->wrap_borders(buffer);
}

} // namespace usfftpp
