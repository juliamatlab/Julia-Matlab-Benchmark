close all; clear; clc;

plotID = {'Matlab', ...
'Julia-1-BLAS-Thread', 'Julia-4-BLAS-Threads','Julia-MKL',...
'Julia-SIMD-1-BLAS-Thread','Julia-SIMD-4-BLAS-Threads','Julia-MKL-SIMD'};
% choose among:
% 'Matlab', 
% 'Julia-1-BLAS-Thread', 'Julia-4-BLAS-Threads','Julia-MKL',
% 'Julia-SIMD-1-BLAS-Thread','Julia-SIMD-4-BLAS-Threads','Julia-MKL-SIMD'

% put the SIMD IDs in the end for correct legend


MatlabAnalysisPlotter(plotID);