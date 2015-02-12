#include <stdio.h>
#include <assert.h>

#include "cuda_runtime.h"
#include <cuda.h>

#include <cusolverDn.h>

/*******************/
/* iDivUp FUNCTION */
/*******************/
extern "C" int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }

/********************/
/* CUDA ERROR CHECK */
/********************/
// --- Credit to http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	  if (abort) { exit(code); }
   }
}

extern "C" void gpuErrchk(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }

/**************************/
/* CUSOLVE ERROR CHECKING */
/**************************/
static const char *_cusolverGetErrorEnum(cusolverStatus_t error)
{
    switch (error)
    {
        case CUSOLVER_STATUS_SUCCESS:
            return "CUSOLVER_SUCCESS";

        case CUSOLVER_STATUS_NOT_INITIALIZED:
            return "CUSOLVER_STATUS_NOT_INITIALIZED";

        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "CUSOLVER_STATUS_ALLOC_FAILED";

        case CUSOLVER_STATUS_INVALID_VALUE:
            return "CUSOLVER_STATUS_INVALID_VALUE";

        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "CUSOLVER_STATUS_ARCH_MISMATCH";

        case CUSOLVER_STATUS_EXECUTION_FAILED:
            return "CUSOLVER_STATUS_EXECUTION_FAILED";

        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "CUSOLVER_STATUS_INTERNAL_ERROR";

        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

    }

    return "<unknown>";
}

inline void __cusolveSafeCall(cusolverStatus_t err, const char *file, const int line)
{
    if(CUSOLVER_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUSOLVE error in file '%s', line %d\n %s\nerror %d: %s\nterminating!\n",__FILE__, __LINE__,err, \
                                _cusolverGetErrorEnum(err)); \
		cudaDeviceReset(); assert(0); \
	}
}

extern "C" void cusolveSafeCall(cusolverStatus_t err) { __cusolveSafeCall(err, __FILE__, __LINE__); }

/*************************/
/* CUBLAS ERROR CHECKING */
/*************************/
static const char *_cublasGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
}

    return "<unknown>";
}

inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
{
    if(CUBLAS_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUBLAS error in file '%s', line %d\n %s\nerror %d: %s\nterminating!\n",__FILE__, __LINE__,err, \
                                _cublasGetErrorEnum(err)); \
		cudaDeviceReset(); assert(0); \
	}
}

extern "C" void cublasSafeCall(cublasStatus_t err) { __cublasSafeCall(err, __FILE__, __LINE__); }
