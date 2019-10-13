close all; clear; clc;

% choose plotID among:
% 'Matlab', ...
% 'Julia-1-BLAS-Thread', 'Julia-4-BLAS-Threads','Julia-MKL',...
% 'Julia-SIMD-1-BLAS-Thread','Julia-SIMD-4-BLAS-Threads','Julia-MKL-SIMD'
% put the SIMD IDs in the end for correct legend

%% Julia vs Matlab
plotID = {'Matlab', ...
'Julia-1-BLAS-Thread','Julia-MKL'};

MatlabAnalysisPlotter(plotID);

%% Julia openBLAS vs Julia MKL
plotID = {'Julia-1-BLAS-Thread', 'Julia-4-BLAS-Threads','Julia-MKL'};

MatlabAnalysisPlotter(plotID);


%% Julia SIMD vs Julia openBLAS
plotID = {'Julia-1-BLAS-Thread','Julia-4-BLAS-Threads',...,
          'Julia-SIMD-1-BLAS-Thread','Julia-SIMD-4-BLAS-Threads'};

MatlabAnalysisPlotter(plotID);

%% Everything

plotID = {'Matlab',... 
'Julia-1-BLAS-Thread', 'Julia-4-BLAS-Threads','Julia-MKL',...
'Julia-SIMD-1-BLAS-Thread','Julia-SIMD-4-BLAS-Threads','Julia-MKL-SIMD'};
MatlabAnalysisPlotter(plotID);

