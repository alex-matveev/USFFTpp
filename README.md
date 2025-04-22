# USFFTpp

USFFTpp is a small C++ library for computing Unequally Spaced Fast Fourier Transform (USFFT), also known as the Nonuniform FFT (NUFFT) or Nonequispaced FFT (NFFT).

This implimlementation is based on the paper by Dutt and Rokhlin, "Fast Fourier transform for nonequspaced data".

To accelerate computations, the code leverages OpenMP-based parallelization across CPU cores.