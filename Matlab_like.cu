#include <thrust/device_vector.h>

#include "Utilities.cuh"

#define DEBUG

/************/
/* LINSPACE */
/************/
template <class T>
T * linspace(const T a, const T b, const unsigned int N) {
	
	T *out_array; gpuErrchk(cudaMalloc((void**)&out_array, N * sizeof(T)));

	T Dx = (b-a)/(T)(N-1);
   
	thrust::device_ptr<T> d = thrust::device_pointer_cast(out_array); 	
	thrust::transform(thrust::make_counting_iterator(a/Dx), thrust::make_counting_iterator((b+static_cast<T>(1))/Dx), thrust::make_constant_iterator(Dx), d, thrust::multiplies<T>());

	gpuErrchk(cudaMemcpy(&out_array[N - 1], &b, sizeof(T), cudaMemcpyHostToDevice));
	
	return out_array;
}

template float  * linspace<float> (const float  a, const float  b, const unsigned int N);
template double * linspace<double>(const double a, const double b, const unsigned int N);

/*******************/
/* MESHGRID KERNEL */
/*******************/
template <class T>
__global__ void meshgrid_kernel(const T * __restrict__ x, const unsigned int Nx, const T * __restrict__ y, const unsigned int Ny, T * __restrict__ X, T * __restrict__ Y) 
{
	unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	if ((tidx < Nx) && (tidy < Ny)) {	
		X[tidy * Nx + tidx] = x[tidx];
		Y[tidy * Nx + tidx] = y[tidy];
	}
}

/************/
/* MESHGRID */
/************/
#define BLOCKSIZE_MESHGRID_X	16
#define BLOCKSIZE_MESHGRID_Y	16

#include <thrust/pair.h>

template <class T>
thrust::pair<T *,T *> meshgrid(const T *x, const unsigned int Nx, const T *y, const unsigned int Ny) {
	
	T *X; gpuErrchk(cudaMalloc((void**)&X, Nx * Ny * sizeof(T)));
	T *Y; gpuErrchk(cudaMalloc((void**)&Y, Nx * Ny * sizeof(T)));

	dim3 BlockSize(BLOCKSIZE_MESHGRID_X, BLOCKSIZE_MESHGRID_Y);
	dim3 GridSize (iDivUp(Nx, BLOCKSIZE_MESHGRID_X), iDivUp(Ny, BLOCKSIZE_MESHGRID_Y));
	
	meshgrid_kernel<<<GridSize, BlockSize>>>(x, Nx, y, Ny, X, Y);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	return thrust::make_pair(X, Y);
}

template thrust::pair<float  *, float  *>  meshgrid<float>  (const float  *, const unsigned int, const float  *, const unsigned int);
template thrust::pair<double *, double *>  meshgrid<double> (const double *, const unsigned int, const double *, const unsigned int);

/*********/
/* COLON */
/*********/
#include <thrust/sequence.h>

template <class T>
T * colon(const T a, const T step, const T b) {
	
	int N = (int)((b - a)/step) + 1;

	T *out_array; gpuErrchk(cudaMalloc((void**)&out_array, N * sizeof(T)));

	thrust::device_ptr<T> d = thrust::device_pointer_cast(out_array); 	

	thrust::sequence(d, d + N, a, step);

	return out_array;
}

template float  * colon<float>  (const float  a, const float  step, const float  b);
template double * colon<double> (const double a, const double step, const double b);

/*****************************/
/* GENERATE SYMMETRIC POINTS */
/*****************************/
template<class T> 
T * generateSymmetricPoints(const T step, const T b) {

	const int N = (int)(b / step) + 1;

	T *d_u;	gpuErrchk(cudaMalloc(&d_u, (2 * N - 1) * sizeof(T)));
	
	T *d_u_temp = colon(static_cast<T>(0), step, b);

	gpuErrchk(cudaMemcpy(d_u + N - 1, d_u_temp, N * sizeof(T), cudaMemcpyDeviceToDevice));
	
 	reverseArray(d_u_temp + 1, d_u, N - 1, static_cast<T>(-1));

	gpuErrchk(cudaFree(d_u_temp));

	return d_u;
}

template float  * generateSymmetricPoints<float>  (const float , const float );
template double * generateSymmetricPoints<double> (const double, const double);
