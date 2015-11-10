function make()
blas_lib = '/misc/apps/matlab/matlabR2014a/bin/glnxa64/libmwblas.so';
mex('-DDEFINEUNIX','mtimesx.c',blas_lib)
% mex('-DDEFINEUNIX','-largeArrayDims','mtimesx.c',blas_lib)
end