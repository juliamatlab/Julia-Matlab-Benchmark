# Julia-Matlab-Benchmark

This repository is a place for accurate benchmarks between Julia and MATLAB and comparing the two.

Various commonly used operations for Matrix operations, Mathematical calculations, Data Processing, Image processing, Signal processing, and different algorithms are tested.

#[Julia vs Matlab](https://github.com/juliamatlab/Julia-Matlab-Benchmark/blob/master/README.md)

#[Julia openBLAS vs Julia MKL](https://github.com/juliamatlab/Julia-Matlab-Benchmark/blob/master/README-Julia-openBLAS-vs-Julia-MKL.md)

#[Julia SIMD vs Julia openBLAS](https://github.com/juliamatlab/Julia-Matlab-Benchmark/blob/master/README-Julia-SIMD-vs-Julia-openBLAS.md)

#[Everything](https://github.com/juliamatlab/Julia-Matlab-Benchmark/blob/master/README-Everything.md)

## Development and Future
This repository will be extended as more functions are added to the [JuliaMatlab](https://github.com/juliamatlab) repository, which is meant to map all the Matlab functions to Julia native functions.

## Other Features
* Latest Julia language is used (compatible with 1.0.4 and higher).
* Julia + Intel MKL is also tested. (For tutorial to install: https://github.com/aminya/MKL.jl/tree/patch-1)
* For some functions, Julia's multithreading and SIMD is used instead of built-in functions.
* Accurate benchmarking tools are used both in Julia and MATLAB to get an reliable result

#  Julia-SIMD-vs-Julia-openBLAS

## Results


### Matrix Addition

Addition of 2 square matrices where each is multiplied by a scalar.

 * MATLAB Code - `mA = (scalarA .* mX) + (scalarB .* mY)`.
 * Julia Code - `mA = (scalarA .* mX) .+ (scalarB .* mY)` (Using the dot for [Loop Fusion][50]).

![Matrix Addition][02]

### Matrix Multiplication

Multiplication of 2 square matrices after a scalar is added to each.

 * MATLAB Code - `mA = (scalarA + mX) * (scalarB + mY)`.
 * Julia Code - `mA = (scalarA .+ mX) * (scalarB .+ mY)` (Using the dot for [Loop Fusion][50]).

![Matrix Multiplication][03]


### Element Wise Operations
Set of operations which are element wise.

 * MATLAB Code - `mD = abs(mA) + sin(mB);`, `mE = exp(-(mA .^ 2));` and `mF = (-mB + sqrt((mB .^ 2) - (4 .* mA .* mC))) ./ (2 .* mA);`.
 * Julia Code - `mD = abs.(mA) .+ sin.(mB);`, `mE = exp.(-(mA .^ 2));` and `mF = (-mB .+ sqrt.((mB .^ 2) .- (4 .* mA .* mC))) ./ (2 .* mA);` (Using the dot for [Loop Fusion][50]).

![Element Wise Operations][06]

## How to Run
Download repository. Or add the package in Julia:
```julia
] add https://github.com/juliamatlab/Julia-Matlab-Benchmark
```
### Run the Benchmark - Julia
* From console:

  ```julia
  include("JuliaMain.jl");
  ```

### Run the Benchmark - MATLAB
* From MATLAB command line :

  ```
  MatlabMain
  ```

### Run The Analysis In MATLAB
 * From MATLAB command line

 ```MatlabAnalysisMain```.
 * Images of the performance test will be created and displayed.

### Run The Analysis In Julia
 * From Julia command line

 ```include("JuliaAnalysisMain.jl");```.
 * Images of the performance test will be created and displayed.

## To Do:
 * This repository will be extended as more functions are added to the [MatlabCompat](https://github.com/aminya/MatlabCompat.jl) repository, which is meant to map all the Matlab functions to Julia native functions

 * Check if Julia code is efficient. using https://github.com/JunoLab/Traceur.jl and https://docs.julialang.org/en/v1/manual/performance-tips/index.html

 * Add Python (NumPy): Code has been converted from MATLAB to python using smop. Still needs working https://github.com/aminya/smop
 * Add Octave.

## Discourse Discussion Forum:
coming soon



## System Configuration
 * System Model - Dell Latitude 5590
 https://www.dell.com/en-ca/work/shop/dell-tablets/latitude-5590/spd/latitude-15-5590-laptop
 * CPU - Intel(R) Core(TM) i5-8250U @ 1.6 [GHz] 1800 Mhz, 4 Cores, 8 Logical Processors.
 * Memory - 1x8GB DDR4 2400MHz Non-ECC
 * Windows 10 Professional 64 Bit
 * WORD_SIZE: 64


 * MATLAB R2018b.
    * BLAS Version (`version -blas`) - `Intel(R) Math Kernel Library Version 2018.0.1 Product Build 20171007 for Intel(R) 64 architecture applications, CNR branch AVX2`
    * LAPACK Version (`version -lapack`) - `Intel(R) Math Kernel Library Version 2018.0.1 Product Build 20171007 for Intel(R) 64 architecture applications CNR branch AVX2  Linear Algebra Package Version 3.7.0`

Two version of Julia was used:

 * JuliaMKL: Julia 1.4.0 + MKL.
     * Julia Version (`versioninfo()`) - `Julia VersionVersion 1.4.0-DEV.233 Commit 32e3c9ea36 (2019-10-02 12:28 UTC)`;
     * BLAS Version - `LinearAlgebra.BLAS.vendor(): Intel MKL `.  For tutorial to install https://github.com/JuliaComputing/MKL.jl 
     * LAPACK Version - `libopenblas64_`.
     * LIBM Version - `libopenlibm`.
     * LLVM Version - `libLLVM-6.0.1 (ORCJIT, skylake)`.
     * JULIA_NUM_THREADS = 1. This number of threads is different from BLAS threads. BLAS threads is changed in the code by `BLAS.set_num_threads(1)` and `BLAS.set_num_threads(4)`

 * Julia: Julia 1.4.0
     * Julia Version (`versioninfo()`) - `Julia VersionVersion 1.4.0-DEV.233 Commit 32e3c9ea36 (2019-10-02 12:28 UTC)`;
     * BLAS Version - `LinearAlgebra.BLAS.vendor(): openBlas64 `.
     * LAPACK Version - `libopenblas64_`.
     * LIBM Version - `libopenlibm`.
     * LLVM Version - `libLLVM-6.0.1 (ORCJIT, skylake)`.
     * JULIA_NUM_THREADS = 1. This number of threads is different from BLAS threads. BLAS threads is changed in the code by `BLAS.set_num_threads(1)` and `BLAS.set_num_threads(4)`

  [01]: https://github.com/juliamatlab/Julia-Matlab-Benchmark/raw/master/Figures/Julia-1-BLAS-Thread_Julia-4-BLAS-Threads_Julia-SIMD-1-BLAS-Thread_Julia-SIMD-4-BLAS-Threads/Figure1.png
  [02]: https://github.com/juliamatlab/Julia-Matlab-Benchmark/raw/master/Figures/Julia-1-BLAS-Thread_Julia-4-BLAS-Threads_Julia-SIMD-1-BLAS-Thread_Julia-SIMD-4-BLAS-Threads/Figure2.png
  [03]: https://github.com/juliamatlab/Julia-Matlab-Benchmark/raw/master/Figures/Julia-1-BLAS-Thread_Julia-4-BLAS-Threads_Julia-SIMD-1-BLAS-Thread_Julia-SIMD-4-BLAS-Threads/Figure3.png
  [04]: https://github.com/juliamatlab/Julia-Matlab-Benchmark/raw/master/Figures/Julia-1-BLAS-Thread_Julia-4-BLAS-Threads_Julia-SIMD-1-BLAS-Thread_Julia-SIMD-4-BLAS-Threads/Figure4.png
  [05]: https://github.com/juliamatlab/Julia-Matlab-Benchmark/raw/master/Figures/Julia-1-BLAS-Thread_Julia-4-BLAS-Threads_Julia-SIMD-1-BLAS-Thread_Julia-SIMD-4-BLAS-Threads/Figure5.png
  [06]: https://github.com/juliamatlab/Julia-Matlab-Benchmark/raw/master/Figures/Julia-1-BLAS-Thread_Julia-4-BLAS-Threads_Julia-SIMD-1-BLAS-Thread_Julia-SIMD-4-BLAS-Threads/Figure6.png
  [07]: https://github.com/juliamatlab/Julia-Matlab-Benchmark/raw/master/Figures/Julia-1-BLAS-Thread_Julia-4-BLAS-Threads_Julia-SIMD-1-BLAS-Thread_Julia-SIMD-4-BLAS-Threads/Figure7.png
  [08]: https://github.com/juliamatlab/Julia-Matlab-Benchmark/raw/master/Figures/Julia-1-BLAS-Thread_Julia-4-BLAS-Threads_Julia-SIMD-1-BLAS-Thread_Julia-SIMD-4-BLAS-Threads/Figure8.png
  [09]: https://github.com/juliamatlab/Julia-Matlab-Benchmark/raw/master/Figures/Julia-1-BLAS-Thread_Julia-4-BLAS-Threads_Julia-SIMD-1-BLAS-Thread_Julia-SIMD-4-BLAS-Threads/Figure9.png
  [10]: https://github.com/juliamatlab/Julia-Matlab-Benchmark/raw/master/Figures/Julia-1-BLAS-Thread_Julia-4-BLAS-Threads_Julia-SIMD-1-BLAS-Thread_Julia-SIMD-4-BLAS-Threads/Figure10.png
  [11]: https://github.com/juliamatlab/Julia-Matlab-Benchmark/raw/master/Figures/Julia-1-BLAS-Thread_Julia-4-BLAS-Threads_Julia-SIMD-1-BLAS-Thread_Julia-SIMD-4-BLAS-Threads/Figure11.png
  [12]: https://github.com/juliamatlab/Julia-Matlab-Benchmark/raw/master/Figures/Julia-1-BLAS-Thread_Julia-4-BLAS-Threads_Julia-SIMD-1-BLAS-Thread_Julia-SIMD-4-BLAS-Threads/Figure12.png
  [13]: https://github.com/juliamatlab/Julia-Matlab-Benchmark/raw/master/Figures/Julia-1-BLAS-Thread_Julia-4-BLAS-Threads_Julia-SIMD-1-BLAS-Thread_Julia-SIMD-4-BLAS-Threads/Figure13.png
  [14]: https://github.com/juliamatlab/Julia-Matlab-Benchmark/raw/master/Figures/Julia-1-BLAS-Thread_Julia-4-BLAS-Threads_Julia-SIMD-1-BLAS-Thread_Julia-SIMD-4-BLAS-Threads/Figure14.png
  [15]: https://github.com/juliamatlab/Julia-Matlab-Benchmark/raw/master/Figures/Julia-1-BLAS-Thread_Julia-4-BLAS-Threads_Julia-SIMD-1-BLAS-Thread_Julia-SIMD-4-BLAS-Threads/Figure15.png
  [16]: https://github.com/juliamatlab/Julia-Matlab-Benchmark/raw/master/Figures/Julia-1-BLAS-Thread_Julia-4-BLAS-Threads_Julia-SIMD-1-BLAS-Thread_Julia-SIMD-4-BLAS-Threads/Figure16.png
  [50]: http://julialang.org/blog/2017/01/moredots

The idea for this repository is taken from https://github.com/aminya/MatlabJuliaMatrixOperationsBenchmark which was a fork from https://github.com/RoyiAvital/MatlabJuliaMatrixOperationsBenchmark

#### By Amin Yahyaabadi
