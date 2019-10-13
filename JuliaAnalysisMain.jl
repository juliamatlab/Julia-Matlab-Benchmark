using DelimitedFiles
using LinearAlgebra
using Plots
gr()
include("JuliaAnalysisPlotter.jl")


const plotID = ["Matlab",
"Julia-1-BLAS-Thread", "Julia-4-BLAS-Threads","Julia-MKL",
"Julia-SIMD-1-BLAS-Thread","Julia-SIMD-4-BLAS-Threads","Julia-MKL-SIMD"]
#=
choose among:
"Matlab",
"Julia-1-BLAS-Thread", "Julia-4-BLAS-Threads","Julia-MKL",
"Julia-SIMD-1-BLAS-Thread","Julia-SIMD-4-BLAS-Threads","Julia-MKL-SIMD"
=#

JuliaAnalysisPlotter(plotID)
