#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include <fftw3.h>

#include "usfftpp.h"

#define _USE_MATH_DEFINES
#include <math.h>

#pragma once

namespace usfftpp {

template <typename T, typename FoldPolicy> void plan<T, 1, FoldPolicy>::reorder_points() {
    m_di = std::make_unique<std::pair<std::size_t, std::size_t>[]>(m_points.size());

#pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < m_points.size(); i++) {
        m_di[i].first = static_cast<std::ptrdiff_t>((m_oversamplingFactor * m_N[0] * m_points[i]) +
                                                    m_oversamplingFactor * m_N[0] / 2);
        m_di[i].second = i;
    }

    std::sort(m_di.get(), m_di.get() + m_points.size());
    std::vector<T> tmp_points;

    tmp_points.reserve(m_points.size());
    for (ptrdiff_t i = 0; i < m_points.size(); i++) {
        tmp_points.push_back(m_points[m_di[i].second]);
    }

    m_points = tmp_points;
}

template <typename T, typename FoldPolicy>
void plan<T, 1, FoldPolicy>::fill_first_occurrence_array() {
    std::ptrdiff_t msize = (m_oversamplingFactor * m_N[0]);
    std::ptrdiff_t pos = 0;

    while (pos < m_points.size()) {
        std::ptrdiff_t ind = m_di[pos].first;
        m_firstOccurrenceArray[ind] = pos;
        while (pos < m_points.size() && ind == m_di[pos].first) {
            pos++;
        }
    }

    m_firstOccurrenceArray[msize] = m_points.size();

    for (std::ptrdiff_t i = msize - 1; i >= m_di[0].first + 1; i--) {
        if (m_firstOccurrenceArray[i] == 0) {
            m_firstOccurrenceArray[i] = m_firstOccurrenceArray[i + 1];
        }
    }
}

template <typename T, typename FoldPolicy>
void plan<T, 1, FoldPolicy>::deconvolute(std::complex<T> *data) {
    T a = sqrt(m_lambda) / sqrt(M_PI);
    T b = M_PI * M_PI / (m_N[0] * m_N[0]) / (m_lambda * 4);

#pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < m_N[0]; i++) {
        T coef = a * exp(static_cast<T>((i - m_N[0] / 2) * (i - m_N[0] / 2)) * b);
        data[i] *= coef;
    }
}

template <typename T, typename FoldPolicy>
void plan<T, 1, FoldPolicy>::do_zero_padding(std::complex<T> *in, std::complex<T> *out) {
    memcpy(out + m_strides[0] + (m_N[0] / 2 + m_N[0] % 2) * m_strides[1], in,
           m_N[0] * sizeof(std::complex<T>));
}

template <typename T, typename FoldPolicy>
void plan<T, 1, FoldPolicy>::undo_zero_padding(std::complex<T> *in, std::complex<T> *out) {
    memcpy(out, in + m_strides[0] + (m_N[0] / 2 + m_N[0] % 2) * m_strides[1],
           m_N[0] * sizeof(std::complex<T>));
}

template <typename T> void fftshift(std::complex<T> *data, std::size_t N) {
    auto temp_data = std::make_unique<std::complex<T>[]>(N);
    for (std::size_t i = 0; i < N; i++) {
        auto new_i = (i + N / 2) % N;
        temp_data[new_i] = data[i];
    }

    for (std::size_t i = 0; i < N; i++) {
        data[i] = temp_data[i];
    }
}

template <typename T, typename FoldPolicy>
void plan<T, 1, FoldPolicy>::fft(std::complex<T> *data, fourier_direction direction) {

    const int padded_size[] = {static_cast<int>(m_oversamplingFactor * m_N[0])};

    int sign;
    if (direction == fourier_direction::forward) {
        sign = FFTW_FORWARD;
    } else {
        sign = FFTW_BACKWARD;
    }

    fftshift(data + m_radius, m_oversamplingFactor * m_N[0]);

    if constexpr (std::is_same<T, float>::value) {
        fftwf_plan p = fftwf_plan_dft_1d(padded_size[0], (fftwf_complex *)&data[m_radius],
                                         (fftwf_complex *)&data[m_radius], sign, FFTW_ESTIMATE);

        fftwf_execute(p);
        fftwf_destroy_plan(p);
    } else {
        fftw_plan p = fftw_plan_dft_1d(padded_size[0], (fftw_complex *)&data[m_radius],
                                       (fftw_complex *)&data[m_radius], sign, FFTW_ESTIMATE);

        fftw_execute(p);
        fftw_destroy_plan(p);
    }

    fftshift(data + m_radius, m_oversamplingFactor * m_N[0]);
}

template <typename T, typename FoldPolicy>
void plan<T, 1, FoldPolicy>::fill_borders(std::complex<T> *data) {
    std::ptrdiff_t padded_size[1] = {static_cast<std::ptrdiff_t>(m_oversamplingFactor) * m_N[0]};

#pragma omp parallel for
    for (std::ptrdiff_t i = -m_radius; i < 0; i++) {
        std::ptrdiff_t ind1 = m_strides[0] + m_strides[1] * i;
        std::ptrdiff_t ind2 =
            m_strides[0] + m_strides[1] * ((i + padded_size[0]) % (padded_size[0]));
        data[ind1] = data[ind2];
    }

#pragma omp parallel for
    for (std::ptrdiff_t i = padded_size[0]; i < padded_size[0] + m_radius; i++) {
        std::ptrdiff_t ind1 = m_strides[0] + m_strides[1] * i;
        std::ptrdiff_t ind2 =
            m_strides[0] + m_strides[1] * ((i + padded_size[0]) % (padded_size[0]));
        data[ind1] = data[ind2];
    }
}

template <typename T, typename FoldPolicy>
void plan<T, 1, FoldPolicy>::wrap_borders(std::complex<T> *data) {
    std::ptrdiff_t padded_size[1] = {static_cast<std::ptrdiff_t>(m_oversamplingFactor) * m_N[0]};

#pragma omp parallel for
    for (std::ptrdiff_t i = -m_radius; i < 0; i++) {
        std::ptrdiff_t ind1 = m_strides[0] + m_strides[1] * i;
        std::ptrdiff_t ind2 =
            m_strides[0] + m_strides[1] * ((i + padded_size[0]) % (padded_size[0]));
        data[ind2] += data[ind1];
    }

#pragma omp parallel for
    for (std::ptrdiff_t i = padded_size[0]; i < padded_size[0] + m_radius; i++) {
        std::ptrdiff_t ind1 = m_strides[0] + m_strides[1] * i;
        std::ptrdiff_t ind2 =
            m_strides[0] + m_strides[1] * ((i + padded_size[0]) % (padded_size[0]));
        data[ind2] += data[ind1];
    }
}

template <typename T, typename FoldPolicy> std::size_t plan<T, 1, FoldPolicy>::size_of_buffer() {
    return (m_oversamplingFactor * m_N[0] + 2 * m_radius);
}

template <typename T, typename FoldPolicy>
int plan<T, 1, FoldPolicy>::nonunform_to_uniform_transform(std::complex<T> *in,
                                                           std::complex<T> *out,
                                                           fourier_direction direction) {

    size_t size = size_of_buffer();
    auto buffer = std::make_unique<std::complex<T>[]>(size);

    scatter(in, buffer.get());
    fft(buffer.get(), direction);

    undo_zero_padding(buffer.get(), out);
    deconvolute(out);
    return 0;
}

template <typename T, typename FoldPolicy>
int plan<T, 1, FoldPolicy>::uninform_to_nonuniform_transform(std::complex<T> *in,
                                                             std::complex<T> *out,
                                                             fourier_direction direction) {
    size_t size = size_of_buffer();
    auto buffer = std::make_unique<std::complex<T>[]>(size);

    deconvolute(in);

    do_zero_padding(in, buffer.get());
    fft(buffer.get(), direction);

    gather(out, buffer.get());
    return 0;
}

template <typename T, typename FoldPolicy>
plan<T, 1, FoldPolicy>::plan(std::array<std::ptrdiff_t, 1> N, std::vector<T> &points, T epsilon)
    : m_points(points), m_N(N), m_epsilon(epsilon),
      m_lambda((M_PI * M_PI) / (-(2) * log(m_epsilon))),
      m_radius((std::ptrdiff_t)(sqrt(-log(epsilon) / m_lambda))),
      m_firstOccurrenceArray(std::make_unique<std::ptrdiff_t[]>(2 * m_N[0] + 1)),
      m_strides({static_cast<std::size_t>(m_radius), 1}) {
    reorder_points();
}

} // namespace usfftpp
