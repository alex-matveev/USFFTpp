#include <array>
#include <complex>
#include <memory>
#include <vector>

#pragma once

namespace usfftpp {

namespace detail {
    struct aligned_deleter {
        template <typename T>
        void operator()(T *ptr){
            #ifdef _WIN32
                _aligned_free(ptr);
            #else
                std::free(ptr);
            #endif
        }
    };

    void* aligned_alloc(size_t align, size_t size) {
        #ifdef _WIN32
            return _aligned_malloc(size, align);
        #else
            return std::aligned_alloc(align, size);
        #endif
    }
}

template <std::size_t D> class simple_par_visitor_policy {
  public:
    template <typename T, typename Functor>
    static void run(Functor fun, std::vector<T> &x, std::complex<T> *f, std::complex<T> *scaled);
};

template <> class simple_par_visitor_policy<1> {

  public:
    template <typename T, typename Functor>
    static void run(Functor fun, std::vector<T> &x, std::complex<T> *f, std::complex<T> *scaled) {
#pragma omp parallel
        {
#pragma omp for schedule(dynamic, 2048)
            for (std::ptrdiff_t i = 0; i < x.size(); i++) {
                fun(i, f, scaled);
            }
        }
    }
};

template <> class simple_par_visitor_policy<2> {

  public:
    template <typename T, typename Functor>
    static void run(Functor fun, std::vector<std::tuple<T, T>> &x, std::complex<T> *f,
                    std::complex<T> *scaled, std::size_t stateSize) {
#pragma omp parallel
        {
            auto local_weight_x = std::unique_ptr<T[], detail::aligned_deleter>(static_cast<T*>(detail::aligned_alloc(64, stateSize * sizeof(T))));
            auto local_weight_y = std::unique_ptr<T[], detail::aligned_deleter>(static_cast<T*>(detail::aligned_alloc(64, stateSize * sizeof(T))));

#pragma omp for schedule(dynamic, 64)
            for (std::ptrdiff_t i = 0; i < x.size(); i++) {
                fun(i, f, scaled, local_weight_x.get(), local_weight_y.get());
            }
        }
    }
};

template <std::size_t D> class simple_par_block_visitor_policy {
  public:
    template <typename T, typename Functor>
    static void run(Functor fun, std::vector<T> &x, std::complex<T> *f, std::complex<T> *scaled);
};

template <> class simple_par_block_visitor_policy<1> {

    static const auto block_size = static_cast<const std::ptrdiff_t>(2 * 1024);

  public:
    template <typename T, typename Functor>
    static void run(Functor fun, std::vector<T> &x, std::complex<T> *f,
                    std::array<std::ptrdiff_t, 1> N, const std::ptrdiff_t *FirstOccurences,
                    std::complex<T> *scaled) {

#pragma omp parallel
        {
            for (std::ptrdiff_t p = 0; p < 2; p++) {
#pragma omp for schedule(dynamic, 16)
                for (std::size_t i = p; i <= 2 * N[0] / block_size; i += 2) {

                    std::size_t count_of_points = (2 * N[0] - block_size * i > block_size)
                                                      ? block_size
                                                      : (2 * N[0] - block_size * i);

                    for (std::size_t j = 0; j < count_of_points; j++) {
                        for (std::size_t l = 0; l < FirstOccurences[i * block_size + j + 1] -
                                                        FirstOccurences[i * block_size + j];
                             l++) {

                            std::size_t k = FirstOccurences[i * block_size + j] + l;
                            fun(k, f, scaled);
                        }
                    }
                }
            }
        }
    }
};

template <> class simple_par_block_visitor_policy<2> {

    static const std::ptrdiff_t block_size = 256;

  public:
    template <typename T, typename Functor>
    static void run(Functor fun, std::vector<std::tuple<T, T>> &x, std::complex<T> *f,
                    std::array<std::ptrdiff_t, 2> N, const std::ptrdiff_t *firstOccurrences,
                    std::complex<T> *scaled, std::ptrdiff_t oversamplingFactor,
                    std::ptrdiff_t step_y, std::ptrdiff_t stateSize) {

#pragma omp parallel
        {
            auto local_weight_x = std::unique_ptr<T[], detail::aligned_deleter>(static_cast<T*>(detail::aligned_alloc(64, stateSize * sizeof(T))));
            auto local_weight_y = std::unique_ptr<T[], detail::aligned_deleter>(static_cast<T*>(detail::aligned_alloc(64, stateSize * sizeof(T))));

            for (std::ptrdiff_t group_i = 0; group_i < step_y; group_i++) {
                for (std::ptrdiff_t group_j = 0; group_j < 2; group_j++) {
                    std::ptrdiff_t count_of_slice_i = (2 * N[0] - group_i - 1) / step_y + 1;

#pragma omp for schedule(dynamic, 1)
                    for (std::ptrdiff_t block_i = 0; block_i < count_of_slice_i; block_i++) {
                        for (std::ptrdiff_t block_j = group_j;
                             block_j <= oversamplingFactor * N[1] / block_size; block_j += 2) {

                            std::ptrdiff_t count_of_points;
                            if (oversamplingFactor * N[1] - block_size * block_j > block_size) {
                                count_of_points = block_size;
                            } else {
                                count_of_points =
                                    (oversamplingFactor * N[1] - block_size * block_j);
                            }
                            for (std::ptrdiff_t j = 0; j < count_of_points; j++) {
                                std::ptrdiff_t ind_foa =
                                    (group_i + step_y * block_i) * (oversamplingFactor * N[1]) +
                                    (block_j * block_size + j);
                                for (std::ptrdiff_t l = 0;
                                     l < firstOccurrences[ind_foa + 1] - firstOccurrences[ind_foa];
                                     l++) {
                                    std::ptrdiff_t k = firstOccurrences[ind_foa] + l;
                                    fun(k, f, scaled, local_weight_x.get(), local_weight_y.get());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

} // namespace usfftpp
