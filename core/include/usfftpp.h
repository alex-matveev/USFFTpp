#include <array>
#include <complex>
#include <cstddef>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#pragma once

namespace usfftpp {

enum class fourier_direction { forward, backward };

template <typename T, std::size_t D, typename FoldPolicy> class plan {
  public:
    int nonunform_to_uniform_transform(std::complex<T> *F, std::complex<T> *f, int FourierType);
    int uninform_to_nonuniform_transform(std::complex<T> *F, std::complex<T> *f, int FourierType);
};

template <typename T, typename FoldPolicy> class plan<T, 1, FoldPolicy> {
  protected:
    std::array<std::ptrdiff_t, 1> m_N;
    T m_epsilon;
    T m_lambda;
    std::size_t m_oversamplingFactor = 2;
    std::ptrdiff_t m_radius;
    std::unique_ptr<std::pair<std::size_t, std::size_t>[]> m_di;
    std::vector<T> m_points;
    std::unique_ptr<std::ptrdiff_t[]> m_firstOccurrenceArray;
    std::array<std::size_t, 2> m_strides;

    void do_zero_padding(std::complex<T> *in, std::complex<T> *out);
    void undo_zero_padding(std::complex<T> *in, std::complex<T> *out);

    void fill_borders(std::complex<T> *data);
    void wrap_borders(std::complex<T> *data);

    void reorder_points();
    void fill_first_occurrence_array();

    void gather(std::complex<T> *out, std::complex<T> *buffer) {
        static_cast<FoldPolicy *>(this)->gather(out, buffer);
    }
    void scatter(std::complex<T> *in, std::complex<T> *buffer) {
        static_cast<FoldPolicy *>(this)->scatter(in, buffer);
    }
    size_t size_of_buffer();
    void fft(std::complex<T> *data, fourier_direction direction);
    void deconvolute(std::complex<T> *data);

  public:
    plan(std::array<std::ptrdiff_t, 1> N, std::vector<T> &points, T epsilon);

    int nonunform_to_uniform_transform(std::complex<T> *in, std::complex<T> *out,
                                       fourier_direction direction);
    int uninform_to_nonuniform_transform(std::complex<T> *in, std::complex<T> *out,
                                         fourier_direction direction);
};

template <typename T, typename FoldPolicy> class plan<T, 2, FoldPolicy> {
  protected:
    std::array<std::ptrdiff_t, 2> m_N;
    T m_epsilon;
    T m_lambda;
    std::size_t m_oversamplingFactor = 2;
    std::ptrdiff_t m_radius;
    std::unique_ptr<std::pair<std::size_t, std::size_t>[]> m_di;
    std::vector<std::tuple<T, T>> m_points;
    std::unique_ptr<std::ptrdiff_t[]> m_firstOccurrenceArray;
    std::array<std::size_t, 3> m_strides;

    void fill_borders(std::complex<T> *data);
    void wrap_borders(std::complex<T> *data);
    void fftshift(std::complex<T> *data, std::array<std::size_t, 2> N);

    void reorder_points();
    void fill_first_occurrence_array();

    void do_zero_padding(std::complex<T> *in, std::complex<T> *out);
    void undo_zero_padding(std::complex<T> *in, std::complex<T> *out);

    void gather(std::complex<T> *out, std::complex<T> *buffer) {
        static_cast<FoldPolicy *>(this)->gather(out, buffer);
    }
    void scatter(std::complex<T> *in, std::complex<T> *buffer) {
        static_cast<FoldPolicy *>(this)->scatter(in, buffer);
    }
    size_t size_of_buffer();
    void fft(std::complex<T> *data, fourier_direction direction);
    void deconvolute(std::complex<T> *data);

  public:
    plan(std::array<std::ptrdiff_t, 2> N, std::vector<std::tuple<T, T>> &points, T epsilon);

    int nonunform_to_uniform_transform(std::complex<T> *in, std::complex<T> *out,
                                       fourier_direction direction);
    int uninform_to_nonuniform_transform(std::complex<T> *in, std::complex<T> *out,
                                         fourier_direction direction);
};
} // namespace usfftpp
