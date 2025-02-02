#include "usfftpp.h"

#include "convPolicies.h"
#include "usfft-seq.h"

#include "impl/seq.hpp"
#include "impl/usfft1d.hpp"
#include "impl/usfft2d.hpp"

namespace usfftpp {

template class plan<
    float, 1, seq<float, 1, simple_par_visitor_policy<1>, simple_par_block_visitor_policy<1>>>;
template class plan<
    double, 1, seq<double, 1, simple_par_visitor_policy<1>, simple_par_block_visitor_policy<1>>>;
template class plan<
    float, 2, seq<float, 2, simple_par_visitor_policy<2>, simple_par_block_visitor_policy<2>>>;
template class plan<
    double, 2, seq<double, 2, simple_par_visitor_policy<2>, simple_par_block_visitor_policy<2>>>;

} // namespace usfftpp