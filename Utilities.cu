#include <stdio.h>
#include <assert.h>

#include "cuda_runtime.h"
#include <cuda.h>

#include <cusolverDn.h>

#include "Utilities.cuh"

#define DEBUG

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
		fprintf(stderr, "CUSOLVE error in file '%s', line %Ndims\Nobjs %s\nerror %Ndims: %s\nterminating!\Nobjs",__FILE__, __LINE__,err, \
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
		fprintf(stderr, "CUBLAS error in file '%s', line %Ndims\Nobjs %s\nerror %Ndims: %s\nterminating!\Nobjs",__FILE__, __LINE__,err, \
                                _cublasGetErrorEnum(err)); \
		cudaDeviceReset(); assert(0); \
	}
}

extern "C" void cublasSafeCall(cublasStatus_t err) { __cublasSafeCall(err, __FILE__, __LINE__); }

/************************/
/* REVERSE ARRAY KERNEL */
/************************/
#define BLOCKSIZE_REVERSE	256

// --- Credit to http://www.drdobbs.com/parallel/cuda-supercomputing-for-the-masses-part/208801731?pgno=2
template <class T>
__global__ void reverseArrayKernel(const T * __restrict__ d_in, T * __restrict__ d_out, const int N)
{
	// --- Credit to the simpleTemplates CUDA sample
	SharedMemory<T> smem;
    T* s_data = smem.getPointer();

    const int tid			= blockDim.x * blockIdx.x + threadIdx.x;
	const int id			= threadIdx.x;
	const int offset		= blockDim.x * (blockIdx.x + 1);

	// --- Load one element per thread from device memory and store it *in reversed order* into shared memory
	if (tid < N) s_data[BLOCKSIZE_REVERSE - (id + 1)] = d_in[tid]; 
 
	// --- Block until all threads in the block have written their data to shared memory
	__syncthreads();
 
	// --- Write the data from shared memory in forward order
	if ((N - offset + id) >= 0) d_out[N - offset + id] = s_data[threadIdx.x]; 
}
 
/************************/
/* REVERSE ARRAY KERNEL */
/************************/
template <class T>
void reverseArray(const T * __restrict__ d_in, T * __restrict__ d_out, const int N) {

    reverseArrayKernel<<<iDivUp(N, BLOCKSIZE_REVERSE), BLOCKSIZE_REVERSE, BLOCKSIZE_REVERSE * sizeof(T)>>>(d_in, d_out, N);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

}

template void reverseArray<float>  (const float  * __restrict__, float  * __restrict__, const int);
template void reverseArray<double> (const double * __restrict__, double * __restrict__, const int);

/***********************************/
/* REVERSE AND NEGATE ARRAY KERNEL */
/***********************************/
// --- Credit to http://www.drdobbs.com/parallel/cuda-supercomputing-for-the-masses-part/208801731?pgno=2
template <class T>
__global__ void reverseAndNegateArrayKernel(const T * __restrict__ d_in, T * __restrict__ d_out, const int N)
{
	// --- Credit to the simpleTemplates CUDA sample
	SharedMemory<T> smem;
    T* s_data = smem.getPointer();

    const int tid			= blockDim.x * blockIdx.x + threadIdx.x;
	const int id			= threadIdx.x;
	const int offset		= blockDim.x * (blockIdx.x + 1);

	// --- Load one element per thread from device memory and store it *in reversed order* into shared memory
	if (tid < N) s_data[BLOCKSIZE_REVERSE - (id + 1)] = -d_in[tid]; 
 
	// --- Block until all threads in the block have written their data to shared memory
	__syncthreads();
 
	// --- Write the data from shared memory in forward order
	if ((N - offset + id) >= 0) d_out[N - offset + id] = s_data[threadIdx.x]; 
}
 
/************************/
/* REVERSE ARRAY KERNEL */
/************************/
template <class T>
void reverseAndNegateArray(const T * __restrict__ d_in, T * __restrict__ d_out, const int N) {

    reverseAndNegateArrayKernel<<<iDivUp(N, BLOCKSIZE_REVERSE), BLOCKSIZE_REVERSE, BLOCKSIZE_REVERSE * sizeof(T)>>>(d_in, d_out, N);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

}

template void reverseAndNegateArray<float>  (const float  * __restrict__, float  * __restrict__, const int);
template void reverseAndNegateArray<double> (const double * __restrict__, double * __restrict__, const int);
