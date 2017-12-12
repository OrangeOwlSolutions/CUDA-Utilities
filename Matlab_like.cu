#include <stdlib.h>

#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#include "Utilities.cuh"

#define DEBUG

/******************/
/* LINSPACE - GPU */
/******************/
template <class T>
T * d_linspace(const T a, const T b, const unsigned int N) {

	T *out_array; gpuErrchk(cudaMalloc((void**)&out_array, N * sizeof(T)));

	T Dx = (b - a) / (T)(N - 1);

	thrust::device_ptr<T> d = thrust::device_pointer_cast(out_array);
	thrust::transform(thrust::make_counting_iterator(a / Dx), thrust::make_counting_iterator((b + static_cast<T>(1)) / Dx), thrust::make_constant_iterator(Dx), d, thrust::multiplies<T>());

	gpuErrchk(cudaMemcpy(&out_array[N - 1], &b, sizeof(T), cudaMemcpyHostToDevice));

	return out_array;
}

template float  * d_linspace<float>(const float  a, const float  b, const unsigned int N);
template double * d_linspace<double>(const double a, const double b, const unsigned int N);

/******************/
/* LINSPACE - CPU */
/******************/
template <class T>
T * h_linspace(const T a, const T b, const unsigned int N) {

	T *out_array = (T *)malloc(N * sizeof(T));

	T Dx = (b - a) / (T)(N - 1);

	//thrust::device_ptr<T> d = thrust::device_pointer_cast(out_array); 	
	//thrust::transform(thrust::host, thrust::make_counting_iterator(a/Dx), thrust::make_counting_iterator((b+static_cast<T>(1))/Dx), thrust::make_constant_iterator(Dx), d, thrust::multiplies<T>());

	//memcpy(&out_array[N - 1], &b, sizeof(T));

	T temp = a / Dx;
	for (int i = 0; i < N; i++) out_array[i] = (temp + i) * Dx;

	return out_array;
}

template float  * h_linspace<float>(const float  a, const float  b, const unsigned int N);
template double * h_linspace<double>(const double a, const double b, const unsigned int N);

/***************/
/* ZEROS - CPU */
/***************/
template <class T> T * h_zeros(const int M, const int N) { T *out_array = (T *)calloc(M * N, sizeof(T)); return out_array;  }

template float  * h_zeros<float> (const int M, const int N);
template double * h_zeros<double>(const int M, const int N);

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

/******************/
/* MESHGRID - GPU */
/******************/
#define BLOCKSIZE_MESHGRID_X	16
#define BLOCKSIZE_MESHGRID_Y	16

template <class T>
thrust::pair<T *, T *> d_meshgrid(const T *x, const unsigned int Nx, const T *y, const unsigned int Ny) {

	T *X; gpuErrchk(cudaMalloc((void**)&X, Nx * Ny * sizeof(T)));
	T *Y; gpuErrchk(cudaMalloc((void**)&Y, Nx * Ny * sizeof(T)));

	dim3 BlockSize(BLOCKSIZE_MESHGRID_X, BLOCKSIZE_MESHGRID_Y);
	dim3 GridSize(iDivUp(Nx, BLOCKSIZE_MESHGRID_X), iDivUp(Ny, BLOCKSIZE_MESHGRID_Y));

	meshgrid_kernel << <GridSize, BlockSize >> >(x, Nx, y, Ny, X, Y);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	return thrust::make_pair(X, Y);
}

template thrust::pair<float  *, float  *>  d_meshgrid<float>(const float  *, const unsigned int, const float  *, const unsigned int);
template thrust::pair<double *, double *>  d_meshgrid<double>(const double *, const unsigned int, const double *, const unsigned int);

/******************/
/* MESHGRID - CPU */
/******************/
template <class T>
thrust::pair<T *, T *> h_meshgrid(const T *x, const unsigned int Nx, const T *y, const unsigned int Ny) {

	T *X = (T *)malloc(Nx * Ny * sizeof(T));
	T *Y = (T *)malloc(Nx * Ny * sizeof(T));

	for (int j = 0; j < Ny; j++)
		for (int i = 0; i < Nx; i++) {
		X[j * Nx + i] = x[i];
		Y[j * Nx + i] = y[j];
		}

	return thrust::make_pair(X, Y);
}

template thrust::pair<float  *, float  *>  h_meshgrid<float>(const float  *, const unsigned int, const float  *, const unsigned int);
template thrust::pair<double *, double *>  h_meshgrid<double>(const double *, const unsigned int, const double *, const unsigned int);

/***************/
/* COLON - GPU */
/***************/
template <class T>
T * d_colon(const T a, const T step, const T b) {

	int N = (int)((b - a) / step) + 1;

	T *out_array; gpuErrchk(cudaMalloc((void**)&out_array, N * sizeof(T)));

	thrust::device_ptr<T> d = thrust::device_pointer_cast(out_array);

	thrust::sequence(d, d + N, a, step);

	return out_array;
}

template float  * d_colon<float>(const float  a, const float  step, const float  b);
template double * d_colon<double>(const double a, const double step, const double b);

/***************/
/* COLON - CPU */
/***************/
template <class T>
T * h_colon(const T a, const T step, const T b) {

	int N = (int)((b - a) / step) + 1;

	T *out_array = (T *)malloc(N * sizeof(T));

	thrust::device_ptr<T> d = thrust::device_pointer_cast(out_array);

	thrust::sequence(thrust::host, d, d + N, a, step);

	return out_array;
}

template float  * h_colon<float>(const float  a, const float  step, const float  b);
template double * h_colon<double>(const double a, const double step, const double b);

/*****************************/
/* GENERATE SYMMETRIC POINTS */
/*****************************/
template<class T>
T * generateSymmetricPoints(const T step, const T b) {

	const int N = (int)(b / step) + 1;

	T *d_u;	gpuErrchk(cudaMalloc(&d_u, (2 * N - 1) * sizeof(T)));

	T *d_u_temp = d_colon(static_cast<T>(0), step, b);

	gpuErrchk(cudaMemcpy(d_u + N - 1, d_u_temp, N * sizeof(T), cudaMemcpyDeviceToDevice));

	reverseArray(d_u_temp + 1, d_u, N - 1, static_cast<T>(-1));

	gpuErrchk(cudaFree(d_u_temp));

	return d_u;
}

template float  * generateSymmetricPoints<float>(const float, const float);
template double * generateSymmetricPoints<double>(const double, const double);
