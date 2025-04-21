#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <tuple>
#include <utility>
#include <vector>

#include <fftw3.h>

#include "usfftpp.h"

#pragma once

namespace usfftpp {

template <typename T, typename FoldPolicy> void plan<T, 2, FoldPolicy>::reorder_points() {

    m_di = std::make_unique<std::pair<std::size_t, std::size_t>[]>(m_points.size());

#pragma omp parallel for
    for (std::size_t i = 0; i < m_points.size(); i++) {
        std::size_t indx = (m_oversamplingFactor * m_N[1] * std::get<1>(m_points[i])) +
                           m_oversamplingFactor * m_N[1] / 2;
        std::size_t indy = (m_oversamplingFactor * m_N[0] * std::get<0>(m_points[i])) +
                           m_oversamplingFactor * m_N[0] / 2;
        m_di[i].first = indx + m_oversamplingFactor * m_N[1] * indy;
        m_di[i].second = i;
    }

    std::sort(m_di.get(), m_di.get() + m_points.size());
    std::vector<std::tuple<T, T>> tmp_points;

    tmp_points.reserve(m_points.size());
    for (ptrdiff_t i = 0; i < m_points.size(); i++) {
        tmp_points.push_back(m_points[m_di[i].second]);
    }

    m_points = tmp_points;
}

template <typename T, typename FoldPolicy>
void plan<T, 2, FoldPolicy>::fill_first_occurrence_array() {
    std::ptrdiff_t pos = 0;
    while (pos < m_points.size()) {
        std::ptrdiff_t ind = m_di[pos].first;
        m_firstOccurrenceArray[ind] = pos;
        while (pos < m_points.size() && ind == m_di[pos].first) {
            pos++;
        }
    }
    m_firstOccurrenceArray[(m_oversamplingFactor * m_N[0]) * (m_oversamplingFactor * m_N[1])] =
        m_points.size();
    for (std::ptrdiff_t i = (m_oversamplingFactor * m_N[0]) * (m_oversamplingFactor * m_N[1]) - 1;
         i >= m_di[0].first + 1; i--) {
        if (m_firstOccurrenceArray[i] == 0) {
            m_firstOccurrenceArray[i] = m_firstOccurrenceArray[i + 1];
        }
    }
}

template <typename T, typename FoldPolicy>
void plan<T, 2, FoldPolicy>::deconvolute(std::complex<T> *data) {

    std::ptrdiff_t strides[3] = {0, m_N[1], 1};

#pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < m_N[0]; i++) {
        for (std::ptrdiff_t j = 0; j < m_N[1]; j++) {
            T coef = m_lambda / M_PI *
                     exp(M_PI * M_PI *
                         ((j - m_N[1] / 2) * (j - m_N[1] / 2) / (T)(m_N[1] * m_N[1]) +
                          (i - m_N[0] / 2) * (i - m_N[0] / 2) / (T)(m_N[0] * m_N[0])) /
                         (m_lambda * 4));
            data[strides[0] + i * strides[1] + j * strides[2]] *= coef;
        }
    }
}

template <typename T, typename FoldPolicy>
void plan<T, 2, FoldPolicy>::do_zero_padding(std::complex<T> *in, std::complex<T> *out) {

#pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < m_N[0]; i++) {
        for (std::ptrdiff_t j = 0; j < m_N[1]; j++) {
            out[m_strides[0] + (i + m_N[0] / 2 + m_N[0] % 2) * m_strides[1] +
                (j + m_N[1] / 2 + m_N[1] % 2) * m_strides[2]] = in[i * m_N[1] + j];
        }
    }
}

template <typename T, typename FoldPolicy>
void plan<T, 2, FoldPolicy>::undo_zero_padding(std::complex<T> *in, std::complex<T> *out) {

    for (std::ptrdiff_t i = 0; i < m_N[0]; i++) {
        for (std::ptrdiff_t j = 0; j < m_N[1]; j++) {
            out[i * m_N[1] + j] = in[m_strides[0] + (i + m_N[0] / 2 + m_N[0] % 2) * m_strides[1] +
                                     (j + m_N[1] / 2 + m_N[1] % 2) * m_strides[2]];
        }
    }
}

template <typename T, typename FoldPolicy>
void plan<T, 2, FoldPolicy>::fftshift(std::complex<T> *data, std::array<std::size_t, 2> N) {
    auto temp_data = std::make_unique<std::complex<T>[]>(N[0] * N[1]);
    for (int i = 0; i < N[0]; i++) {
        for (int j = 0; j < N[1]; j++) {
            auto new_i = (i + N[0] / 2) % N[0];
            auto new_j = (j + N[1] / 2) % N[1];

            temp_data[new_i * N[1] + new_j] =
                data[m_strides[0] + m_strides[1] * i + m_strides[2] * j];
        }
    }

    for (int i = 0; i < N[0]; i++) {
        for (int j = 0; j < N[1]; j++) {
            data[m_strides[0] + m_strides[1] * i + m_strides[2] * j] = temp_data[i * N[1] + j];
        }
    }
}

template <typename T, typename FoldPolicy>
void plan<T, 2, FoldPolicy>::fft(std::complex<T> *data, fourier_direction direction) {
    const std::array<int, 2> doubled_n = {static_cast<int>(m_oversamplingFactor * m_N[0]),
                                          static_cast<int>(m_oversamplingFactor * m_N[1])};
    const std::array<std::size_t, 2> doubled_n2 = {(m_oversamplingFactor * m_N[0]),
                                                   (m_oversamplingFactor * m_N[1])};
    const std::array<int, 2> inembed = {
        static_cast<int>(m_oversamplingFactor * (m_N[0] + m_radius)),
        static_cast<int>(m_oversamplingFactor * (m_N[1] + m_radius))};
    const std::array<int, 2> onembed = {
        static_cast<int>(m_oversamplingFactor * (m_N[0] + m_radius)),
        static_cast<int>(m_oversamplingFactor * (m_N[1] + m_radius))};

    int sign;
    if (direction == fourier_direction::forward) {
        sign = FFTW_FORWARD;
    } else {
        sign = FFTW_BACKWARD;
    }

    fftshift(data, doubled_n2);

    if constexpr (std::is_same<T, float>::value) {
        fftwf_plan p = fftwf_plan_many_dft(
            2, doubled_n.data(), 1,
            (fftwf_complex *)(data + m_radius * (m_strides[1] + m_strides[2])), inembed.data(), 1,
            0, (fftwf_complex *)(data + m_radius * (m_strides[1] + m_strides[2])), onembed.data(),
            1, 0, sign, FFTW_ESTIMATE);

        fftwf_execute(p);
        fftwf_destroy_plan(p);
    } else {
        fftw_plan p = fftw_plan_many_dft(
            2, doubled_n.data(), 1,
            (fftw_complex *)(data + m_radius * (m_strides[1] + m_strides[2])), inembed.data(), 1, 0,
            (fftw_complex *)(data + m_radius * (m_strides[1] + m_strides[2])), onembed.data(), 1, 0,
            sign, FFTW_ESTIMATE);

        fftw_execute(p);
        fftw_destroy_plan(p);
    }

    fftshift(data, doubled_n2);
}

template <typename T, typename FoldPolicy>
void plan<T, 2, FoldPolicy>::fill_borders(std::complex<T> *data) {

#pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < m_oversamplingFactor * m_N[0]; i++) {
        for (std::ptrdiff_t j = -m_radius; j < 0; j++) {
            std::ptrdiff_t ind1 = m_strides[0] + m_strides[1] * i + m_strides[2] * j;
            std::ptrdiff_t ind2 =
                m_strides[0] +
                m_strides[1] *
                    ((i + m_oversamplingFactor * m_N[0]) % (m_oversamplingFactor * m_N[0])) +
                m_strides[2] *
                    ((j + m_oversamplingFactor * m_N[1]) % (m_oversamplingFactor * m_N[1]));
            data[ind1] = data[ind2];
        }
        for (std::ptrdiff_t j = m_oversamplingFactor * m_N[1];
             j < m_oversamplingFactor * m_N[1] + m_radius; j++) {
            std::ptrdiff_t ind1 = m_strides[0] + m_strides[1] * i + m_strides[2] * j;
            std::ptrdiff_t ind2 =
                m_strides[0] +
                m_strides[1] *
                    ((i + m_oversamplingFactor * m_N[0]) % (m_oversamplingFactor * m_N[0])) +
                m_strides[2] *
                    ((j + m_oversamplingFactor * m_N[1]) % (m_oversamplingFactor * m_N[1]));
            data[ind1] = data[ind2];
        }
    }

#pragma omp parallel for
    for (std::ptrdiff_t i = -m_radius; i < 0; i++) {
        for (std::ptrdiff_t j = -m_radius;
             j < (std::ptrdiff_t)(m_oversamplingFactor * m_N[1]) + m_radius; j++) {
            std::ptrdiff_t ind1 = m_strides[0] + m_strides[1] * i + m_strides[2] * j;
            std::ptrdiff_t ind2 =
                m_strides[0] +
                m_strides[1] *
                    ((i + m_oversamplingFactor * m_N[0]) % (m_oversamplingFactor * m_N[0])) +
                m_strides[2] *
                    ((j + m_oversamplingFactor * m_N[1]) % (m_oversamplingFactor * m_N[1]));
            data[ind1] = data[ind2];
        }
    }

#pragma omp parallel for
    for (std::ptrdiff_t i = m_oversamplingFactor * m_N[0];
         i < m_oversamplingFactor * m_N[0] + m_radius; i++) {
        for (std::ptrdiff_t j = -m_radius;
             j < (std::ptrdiff_t)(m_oversamplingFactor * m_N[1]) + m_radius; j++) {
            std::ptrdiff_t ind1 = m_strides[0] + m_strides[1] * i + m_strides[2] * j;
            std::ptrdiff_t ind2 =
                m_strides[0] +
                m_strides[1] *
                    ((i + m_oversamplingFactor * m_N[0]) % (m_oversamplingFactor * m_N[0])) +
                m_strides[2] *
                    ((j + m_oversamplingFactor * m_N[1]) % (m_oversamplingFactor * m_N[1]));
            data[ind1] = data[ind2];
        }
    }
}

template <typename T, typename FoldPolicy>
void plan<T, 2, FoldPolicy>::wrap_borders(std::complex<T> *data) {

#pragma omp parallel for
    for (std::ptrdiff_t i = -m_radius; i < 0; i++) {
        for (std::ptrdiff_t j = -m_radius;
             j < (std::ptrdiff_t)(m_oversamplingFactor * m_N[1]) + m_radius; j++) {
            std::ptrdiff_t ind1 = m_strides[0] + m_strides[1] * i + m_strides[2] * j;
            std::ptrdiff_t ind2 =
                m_strides[0] +
                m_strides[1] *
                    ((i + m_oversamplingFactor * m_N[0]) % (m_oversamplingFactor * m_N[0])) +
                m_strides[2] *
                    ((j + m_oversamplingFactor * m_N[1]) % (m_oversamplingFactor * m_N[1]));
            data[ind2] += data[ind1];
        }
    }

#pragma omp parallel for
    for (std::ptrdiff_t i = m_oversamplingFactor * m_N[0];
         i < m_oversamplingFactor * m_N[0] + m_radius; i++) {
        for (std::ptrdiff_t j = -m_radius;
             j < (std::ptrdiff_t)(m_oversamplingFactor * m_N[1]) + m_radius; j++) {
            std::ptrdiff_t ind1 = m_strides[0] + m_strides[1] * i + m_strides[2] * j;
            std::ptrdiff_t ind2 =
                m_strides[0] +
                m_strides[1] *
                    ((i + m_oversamplingFactor * m_N[0]) % (m_oversamplingFactor * m_N[0])) +
                m_strides[2] *
                    ((j + m_oversamplingFactor * m_N[1]) % (m_oversamplingFactor * m_N[1]));
            data[ind2] += data[ind1];
        }
    }

#pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < m_oversamplingFactor * m_N[0]; i++) {
        for (std::ptrdiff_t j = -m_radius; j < 0; j++) {
            std::ptrdiff_t ind1 = m_strides[0] + m_strides[1] * i + m_strides[2] * j;
            std::ptrdiff_t ind2 =
                m_strides[0] +
                m_strides[1] *
                    ((i + m_oversamplingFactor * m_N[0]) % (m_oversamplingFactor * m_N[0])) +
                m_strides[2] *
                    ((j + m_oversamplingFactor * m_N[1]) % (m_oversamplingFactor * m_N[1]));
            data[ind2] += data[ind1];
        }
        for (std::ptrdiff_t j = m_oversamplingFactor * m_N[1];
             j < m_oversamplingFactor * m_N[1] + m_radius; j++) {
            std::ptrdiff_t ind1 = m_strides[0] + m_strides[1] * i + m_strides[2] * j;
            std::ptrdiff_t ind2 =
                m_strides[0] +
                m_strides[1] *
                    ((i + m_oversamplingFactor * m_N[0]) % (m_oversamplingFactor * m_N[0])) +
                m_strides[2] *
                    ((j + m_oversamplingFactor * m_N[1]) % (m_oversamplingFactor * m_N[1]));
            data[ind2] += data[ind1];
        }
    }
}

template <typename T, typename FoldPolicy> std::size_t plan<T, 2, FoldPolicy>::size_of_buffer() {
    return (m_oversamplingFactor * m_N[0] + 2 * m_radius) *
           (m_oversamplingFactor * m_N[1] + 2 * m_radius);
}

template <typename T, typename FoldPolicy>
int plan<T, 2, FoldPolicy>::nonunform_to_uniform_transform(std::complex<T> *in,
                                                           std::complex<T> *out,
                                                           fourier_direction direction) {

    size_t size = size_of_buffer();
    auto buffer = std::unique_ptr<std::complex<T>[]>(new std::complex<T>[size]());
    scatter(in, buffer.get());
    fft(buffer.get(), direction);

    undo_zero_padding(buffer.get(), out);
    deconvolute(out);

    return 0;
}

template <typename T, typename FoldPolicy>
int plan<T, 2, FoldPolicy>::uninform_to_nonuniform_transform(std::complex<T> *in,
                                                             std::complex<T> *out,
                                                             fourier_direction direction) {
    size_t size = size_of_buffer();
    auto buffer = std::unique_ptr<std::complex<T>[]>(new std::complex<T>[size]());

    deconvolute(in);

    do_zero_padding(in, buffer.get());

    fft(buffer.get(), direction);
    gather(out, buffer.get());

    return 0;
}

template <typename T, typename FoldPolicy>
plan<T, 2, FoldPolicy>::plan(std::array<std::ptrdiff_t, 2> N, std::vector<std::tuple<T, T>> &points,
                             T epsilon)
    : m_points(points), m_N(N), m_epsilon(epsilon),
      m_lambda((M_PI * M_PI) / (-(2) * (log(epsilon)))),
      m_radius((std::size_t)(sqrt(-log(epsilon) / m_lambda))),
      m_di(std::make_unique<std::pair<std::size_t, std::size_t>[]>(points.size())),
      m_firstOccurrenceArray(std::make_unique<std::ptrdiff_t[]>(
          (m_oversamplingFactor * m_N[0]) * (m_oversamplingFactor * m_N[1]) + 1)),
      m_strides({m_radius * (m_oversamplingFactor * m_N[1] + 2 * m_radius + 1),
                 m_oversamplingFactor * m_N[1] + 2 * m_radius, 1}) {
    reorder_points();
}

} // namespace usfftpp
