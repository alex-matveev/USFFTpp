#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "USFFTpp/convPolicies.h"
#include "USFFTpp/usfft-seq.h"
#include "USFFTpp/usfftpp.h"

namespace py = pybind11;

using Plan1dFloat = usfftpp::plan<
    float,
    1,
    usfftpp::seq<float, 1, usfftpp::simple_par_visitor_policy<1>, usfftpp::simple_par_block_visitor_policy<1>>>;

using Plan1dDouble = usfftpp::plan<
    double,
    1,
    usfftpp::seq<double, 1, usfftpp::simple_par_visitor_policy<1>, usfftpp::simple_par_block_visitor_policy<1>>>;

using Plan2dFloat = usfftpp::plan<
    float,
    2,
    usfftpp::seq<float, 2, usfftpp::simple_par_visitor_policy<2>, usfftpp::simple_par_block_visitor_policy<2>>>;

using Plan2dDouble = usfftpp::plan<
    double,
    2,
    usfftpp::seq<double, 2, usfftpp::simple_par_visitor_policy<2>, usfftpp::simple_par_block_visitor_policy<2>>>;

struct Plan1dWrapper {
    std::unique_ptr<Plan1dFloat> plan_f;
    std::unique_ptr<Plan1dDouble> plan_d;

    bool is_float() const { return static_cast<bool>(plan_f); }
    bool is_double() const { return static_cast<bool>(plan_d); }

    Plan1dWrapper(std::ptrdiff_t N, py::array points, double epsilon) {
        auto buf = points.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("points must be a 1D array");
        }

        auto dtype = points.dtype();

        if (dtype.is(py::dtype::of<float>())) {
            std::vector<float> pts(buf.size);
            auto *data = static_cast<float *>(buf.ptr);
            std::copy(data, data + buf.size, pts.begin());
            std::array<std::ptrdiff_t, 1> N_arr{N};
            plan_f = std::make_unique<Plan1dFloat>(N_arr, pts, static_cast<float>(epsilon));
        } else if (dtype.is(py::dtype::of<double>())) {
            std::vector<double> pts(buf.size);
            auto *data = static_cast<double *>(buf.ptr);
            std::copy(data, data + buf.size, pts.begin());
            std::array<std::ptrdiff_t, 1> N_arr{N};
            plan_d = std::make_unique<Plan1dDouble>(N_arr, pts, epsilon);
        } else {
            throw std::runtime_error("points must have dtype float32 or float64");
        }
    }

    void nonuniform_to_uniform(py::array in, py::array out, usfftpp::fourier_direction direction) {
        auto in_buf = in.request();
        auto out_buf = out.request();

        if (in_buf.ndim != 1 || out_buf.ndim != 1) {
            throw std::runtime_error("in and out must be 1D complex arrays");
        }

        if (is_float()) {
            if (!in.dtype().is(py::dtype::of<std::complex<float>>()) ||
                !out.dtype().is(py::dtype::of<std::complex<float>>())) {
                throw std::runtime_error("for float plan, in/out must be complex64 arrays");
            }

            auto *in_ptr = static_cast<std::complex<float> *>(in_buf.ptr);
            auto *out_ptr = static_cast<std::complex<float> *>(out_buf.ptr);
            plan_f->nonunform_to_uniform_transform(in_ptr, out_ptr, direction);
        } else if (is_double()) {
            if (!in.dtype().is(py::dtype::of<std::complex<double>>()) ||
                !out.dtype().is(py::dtype::of<std::complex<double>>())) {
                throw std::runtime_error("for double plan, in/out must be complex128 arrays");
            }

            auto *in_ptr = static_cast<std::complex<double> *>(in_buf.ptr);
            auto *out_ptr = static_cast<std::complex<double> *>(out_buf.ptr);
            plan_d->nonunform_to_uniform_transform(in_ptr, out_ptr, direction);
        } else {
            throw std::runtime_error("plan is not initialized");
        }
    }

    void uniform_to_nonuniform(py::array in, py::array out, usfftpp::fourier_direction direction) {
        auto in_buf = in.request();
        auto out_buf = out.request();

        if (in_buf.ndim != 1 || out_buf.ndim != 1) {
            throw std::runtime_error("in and out must be 1D complex arrays");
        }

        if (is_float()) {
            if (!in.dtype().is(py::dtype::of<std::complex<float>>()) ||
                !out.dtype().is(py::dtype::of<std::complex<float>>())) {
                throw std::runtime_error("for float plan, in/out must be complex64 arrays");
            }

            auto *in_ptr = static_cast<std::complex<float> *>(in_buf.ptr);
            auto *out_ptr = static_cast<std::complex<float> *>(out_buf.ptr);
            plan_f->uninform_to_nonuniform_transform(in_ptr, out_ptr, direction);
        } else if (is_double()) {
            if (!in.dtype().is(py::dtype::of<std::complex<double>>()) ||
                !out.dtype().is(py::dtype::of<std::complex<double>>())) {
                throw std::runtime_error("for double plan, in/out must be complex128 arrays");
            }

            auto *in_ptr = static_cast<std::complex<double> *>(in_buf.ptr);
            auto *out_ptr = static_cast<std::complex<double> *>(out_buf.ptr);
            plan_d->uninform_to_nonuniform_transform(in_ptr, out_ptr, direction);
        } else {
            throw std::runtime_error("plan is not initialized");
        }
    }
};

struct Plan2dWrapper {
    std::unique_ptr<Plan2dFloat> plan_f;
    std::unique_ptr<Plan2dDouble> plan_d;

    bool is_float() const { return static_cast<bool>(plan_f); }
    bool is_double() const { return static_cast<bool>(plan_d); }

    Plan2dWrapper(std::ptrdiff_t N0, std::ptrdiff_t N1, py::array points, double epsilon) {
        auto buf = points.request();
        if (buf.ndim != 2 || buf.shape[1] != 2) {
            throw std::runtime_error("points must be a 2D array of shape (M, 2)");
        }

        auto dtype = points.dtype();

        if (dtype.is(py::dtype::of<float>())) {
            std::vector<std::tuple<float, float>> pts;
            pts.reserve(static_cast<std::size_t>(buf.shape[0]));

            auto *data = static_cast<float *>(buf.ptr);
            for (ssize_t i = 0; i < buf.shape[0]; ++i) {
                float x = data[2 * i];
                float y = data[2 * i + 1];
                pts.emplace_back(x, y);
            }

            std::array<std::ptrdiff_t, 2> N_arr{N0, N1};
            plan_f = std::make_unique<Plan2dFloat>(N_arr, pts, static_cast<float>(epsilon));
        } else if (dtype.is(py::dtype::of<double>())) {
            std::vector<std::tuple<double, double>> pts;
            pts.reserve(static_cast<std::size_t>(buf.shape[0]));

            auto *data = static_cast<double *>(buf.ptr);
            for (ssize_t i = 0; i < buf.shape[0]; ++i) {
                double x = data[2 * i];
                double y = data[2 * i + 1];
                pts.emplace_back(x, y);
            }

            std::array<std::ptrdiff_t, 2> N_arr{N0, N1};
            plan_d = std::make_unique<Plan2dDouble>(N_arr, pts, epsilon);
        } else {
            throw std::runtime_error("points must have dtype float32 or float64");
        }
    }

    void nonuniform_to_uniform(py::array in, py::array out, usfftpp::fourier_direction direction) {
        auto in_buf = in.request();
        auto out_buf = out.request();

        if (in_buf.ndim != 1 || out_buf.ndim != 2) {
            throw std::runtime_error("in must be 1D and out must be 2D complex array");
        }

        if (is_float()) {
            if (!in.dtype().is(py::dtype::of<std::complex<float>>()) ||
                !out.dtype().is(py::dtype::of<std::complex<float>>())) {
                throw std::runtime_error("for float plan, in/out must be complex64 arrays");
            }

            auto *in_ptr = static_cast<std::complex<float> *>(in_buf.ptr);
            auto *out_ptr = static_cast<std::complex<float> *>(out_buf.ptr);

            plan_f->nonunform_to_uniform_transform(in_ptr, out_ptr, direction);
        } else if (is_double()) {
            if (!in.dtype().is(py::dtype::of<std::complex<double>>()) ||
                !out.dtype().is(py::dtype::of<std::complex<double>>())) {
                throw std::runtime_error("for double plan, in/out must be complex128 arrays");
            }

            auto *in_ptr = static_cast<std::complex<double> *>(in_buf.ptr);
            auto *out_ptr = static_cast<std::complex<double> *>(out_buf.ptr);

            plan_d->nonunform_to_uniform_transform(in_ptr, out_ptr, direction);
        } else {
            throw std::runtime_error("plan is not initialized");
        }
    }

    void uniform_to_nonuniform(py::array in, py::array out, usfftpp::fourier_direction direction) {
        auto in_buf = in.request();
        auto out_buf = out.request();

        if (in_buf.ndim != 2 || out_buf.ndim != 1) {
            throw std::runtime_error("in must be 2D and out must be 1D complex array");
        }

        if (is_float()) {
            if (!in.dtype().is(py::dtype::of<std::complex<float>>()) ||
                !out.dtype().is(py::dtype::of<std::complex<float>>())) {
                throw std::runtime_error("for float plan, in/out must be complex64 arrays");
            }

            auto *in_ptr = static_cast<std::complex<float> *>(in_buf.ptr);
            auto *out_ptr = static_cast<std::complex<float> *>(out_buf.ptr);

            plan_f->uninform_to_nonuniform_transform(in_ptr, out_ptr, direction);
        } else if (is_double()) {
            if (!in.dtype().is(py::dtype::of<std::complex<double>>()) ||
                !out.dtype().is(py::dtype::of<std::complex<double>>())) {
                throw std::runtime_error("for double plan, in/out must be complex128 arrays");
            }

            auto *in_ptr = static_cast<std::complex<double> *>(in_buf.ptr);
            auto *out_ptr = static_cast<std::complex<double> *>(out_buf.ptr);

            plan_d->uninform_to_nonuniform_transform(in_ptr, out_ptr, direction);
        } else {
            throw std::runtime_error("plan is not initialized");
        }
    }
};

PYBIND11_MODULE(_usfftpp, m) {
    m.doc() = "Python bindings for USFFTpp (float and double precision)";

    py::enum_<usfftpp::fourier_direction>(m, "FourierDirection")
        .value("forward", usfftpp::fourier_direction::forward)
        .value("backward", usfftpp::fourier_direction::backward)
        .export_values();

    py::class_<Plan1dWrapper>(m, "Plan1d")
        .def(
            py::init<std::ptrdiff_t, py::array, double>(),
            py::arg("N"),
            py::arg("points"),
            py::arg("epsilon"))
        .def(
            "nonuniform_to_uniform",
            &Plan1dWrapper::nonuniform_to_uniform,
            py::arg("in"),
            py::arg("out"),
            py::arg("direction"))
        .def(
            "uniform_to_nonuniform",
            &Plan1dWrapper::uniform_to_nonuniform,
            py::arg("in"),
            py::arg("out"),
            py::arg("direction"));

    py::class_<Plan2dWrapper>(m, "Plan2d")
        .def(
            py::init<std::ptrdiff_t, std::ptrdiff_t, py::array, double>(),
            py::arg("N0"),
            py::arg("N1"),
            py::arg("points"),
            py::arg("epsilon"))
        .def(
            "nonuniform_to_uniform",
            &Plan2dWrapper::nonuniform_to_uniform,
            py::arg("in"),
            py::arg("out"),
            py::arg("direction"))
        .def(
            "uniform_to_nonuniform",
            &Plan2dWrapper::uniform_to_nonuniform,
            py::arg("in"),
            py::arg("out"),
            py::arg("direction"));
}

