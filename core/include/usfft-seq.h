#include "usfftpp.h"

#pragma once

namespace usfftpp {
template <typename T, std::size_t D, typename GatherVisitorPolicy, typename ScatterVisitorPolicy>
class USFFTPP_API seq : public plan<T, D, seq<T, D, GatherVisitorPolicy, ScatterVisitorPolicy>> {
  public:
    void gather(std::complex<T> *out, std::complex<T> *buffer);
    void scatter(std::complex<T> *in, std::complex<T> *buffer);
};

template <typename T, typename GatherVisitorPolicy, typename ScatterVisitorPolicy>
class USFFTPP_API seq<T, 1, GatherVisitorPolicy, ScatterVisitorPolicy>
    : public plan<T, 1, seq<T, 1, GatherVisitorPolicy, ScatterVisitorPolicy>> {
  public:
    void gather(std::complex<T> *out, std::complex<T> *buffer);
    void scatter(std::complex<T> *in, std::complex<T> *buffer);
};

template <typename T, typename GatherVisitorPolicy, typename ScatterVisitorPolicy>
class USFFTPP_API seq<T, 2, GatherVisitorPolicy, ScatterVisitorPolicy>
    : public plan<T, 2, seq<T, 2, GatherVisitorPolicy, ScatterVisitorPolicy>> {
  public:
    void gather(std::complex<T> *out, std::complex<T> *buffer);
    void scatter(std::complex<T> *in, std::complex<T> *buffer);
};
} // namespace usfftpp
