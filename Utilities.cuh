#ifndef UTILITIES_CUH
#define UTILITIES_CUH

#include <cusolverDn.h>
#include <cublas_v2.h>

extern "C" int iDivUp(int, int);
extern "C" void gpuErrchk(cudaError_t);
extern "C" void cusolveSafeCall(cusolverStatus_t);
extern "C" void cublasSafeCall(cublasStatus_t);

#endif
