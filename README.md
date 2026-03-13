# USFFTpp

USFFTpp is a small C++ library for computing Unequally Spaced Fast Fourier Transform (USFFT), also known as the Nonuniform FFT (NUFFT) or Nonequispaced FFT (NFFT).

This implementation is based on the paper by Dutt and Rokhlin, "Fast Fourier transform for nonequispaced data".

To accelerate computations, the code leverages OpenMP-based parallelization across CPU cores.

## Requirements

- C++20 compiler
- CMake 3.9+
- [FFTW3](http://www.fftw.org/) (double and single precision)
- OpenMP
- For Python bindings: Python 3.8+, pybind11, NumPy

## Building the C++ library

```bash
mkdir build && cd build
cmake .. -DUSFFTPP_BUILD_TESTS=ON -DUSFFTPP_BUILD_PYTHON=ON
cmake --build .
```

Options:

- `USFFTPP_BUILD_TESTS` — build C++ and Python tests (default: ON)
- `USFFTPP_BUILD_PYTHON` — build Python bindings (default: ON)

## Python package

### Install via pip (from repository root)

```bash
pip install .
```

This uses [scikit-build-core](https://scikit-build-core.readthedocs.io/) and builds the C++ library and the `usfftpp` Python extension.

### Build in-tree (for development)

From the project root, after configuring CMake with `USFFTPP_BUILD_PYTHON=ON`:

```bash
cmake --build build --target _usfftpp
```

The extension is placed in `python/usfftpp/`. Run Python with that directory on the path:

```bash
PYTHONPATH=python python -c "import usfftpp; print(usfftpp.Plan1d)"
```

## Python usage

Import the package and use 1D or 2D plans. Float/double is chosen from the dtype of the `points` array.

```python
import numpy as np
import usfftpp

# Direction for the transform
FourierDirection = usfftpp.FourierDirection  # .forward, .backward

# --- 1D ---
N = 1024
points_1d = np.linspace(-0.5, 0.5, 500, dtype=np.float64)  # or np.float32
plan_1d = usfftpp.Plan1d(N, points_1d, epsilon=1e-6)

# Nonuniform -> uniform (e.g. forward)
f_in = np.zeros(points_1d.shape[0], dtype=np.complex128)
f_out = np.zeros(N, dtype=np.complex128)
plan_1d.nonuniform_to_uniform(f_in, f_out, usfftpp.FourierDirection.forward)

# Uniform -> nonuniform (e.g. backward)
plan_1d.uniform_to_nonuniform(f_out, f_in, usfftpp.FourierDirection.backward)

# --- 2D ---
N0, N1 = 128, 128
points_2d = np.random.rand(1000, 2).astype(np.float64) - 0.5  # shape (M, 2)
plan_2d = usfftpp.Plan2d(N0, N1, points_2d, epsilon=1e-6)

f_in_2d = np.zeros(points_2d.shape[0], dtype=np.complex128)
f_out_2d = np.zeros((N0, N1), dtype=np.complex128)
plan_2d.nonuniform_to_uniform(f_in_2d, f_out_2d, usfftpp.FourierDirection.forward)
```

- **Plan1d**: `points` — 1D array of shape `(M,)` (float32 or float64). Input/output arrays are 1D complex (complex64 or complex128 to match the plan).
- **Plan2d**: `points` — 2D array of shape `(M, 2)` (float32 or float64). For `nonuniform_to_uniform`, `in` is 1D length M and `out` is 2D shape `(N0, N1)`; for `uniform_to_nonuniform`, the shapes are reversed.

## Running tests

After building with tests enabled:

```bash
cd build
ctest
```

To run only the Python tests:

```bash
ctest -R python.usfftpp.basic
```

Or run the Python test module directly (with `python` on `PYTHONPATH` so that `usfftpp` is importable):

```bash
PYTHONPATH=python python -m unittest discover -s python/tests -p "test_*.py"
```

## License

See the project license file (e.g. BSD-3-Clause as referenced in `pyproject.toml`).