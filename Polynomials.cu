#include "Utilities.cuh"

// --- The approach could be improved by calculating the coefficients of x^n and then summing up the polynomials using fused multiply add functions

/*********************************/
/* LEGENDRE POLYNOMIAL FUNCTIONS */
/*********************************/
// --- Zeroth order 
template <class T> inline __host__ __device__ T Legendre0(const T& x) { return static_cast<T>(1.0) ; }

// --- First order
template <class T> inline __host__ __device__ T Legendre1(const T& x) { return x ; }

// --- Second order
template <class T> inline __host__ __device__ T Legendre2(const T& x) { return ((static_cast<T>(3.0) * x * x) - static_cast<T>(1.0)) * static_cast<T>(0.5); }

// --- N-th order
template <class T> inline __host__ __device__ T LegendreN(unsigned int n, const T& x) {
	
	if      (n == 0) { return Legendre0<T>(x); }
    else if (n == 1) { return Legendre1<T>(x); }
    else if (n == 2) { return Legendre2<T>(x); }
    
    if (x == static_cast<T>(1.0))  { return static_cast<T>(1.0); }

    if (x == static_cast<T>(-1.0)) { return ((n % 2 == 0) ? static_cast<T>(1.0) : static_cast<T>(-1.0)); }

    if ((x == static_cast<T>(0.0)) && (n % 2)) { return static_cast<T>(0.0); }

    T pnm1(Legendre2<T>(x));
    T pnm2(Legendre1<T>(x));
    T pn(pnm1);

    for (unsigned int l = 3 ; l <= n ; l++) { 
		pn = (((static_cast<T>(2.0) * static_cast<T>(l)) - static_cast<T>(1.0)) * x * pnm1 - ((static_cast<T>(l) - static_cast<T>(1.0)) * pnm2)) / static_cast<T>(l);
		pnm2 = pnm1;
		pnm1 = pn;
    }

    return pn;
}

/*******************************************/
/* LEGENDRE POLYNOMIALS CALCULATION KERNEL */
/*******************************************/
template <class T>
__global__ void generateLegendreKernel(T * __restrict__ d_Leg, const T * __restrict__ d_x, const int maxDegree, const int N) {

	const int tid_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int tid_y = blockDim.y * blockIdx.y + threadIdx.y;

	if ((tid_x < N) && (tid_y < maxDegree)) d_Leg[tid_y * N + tid_x] = LegendreN(tid_y, d_x[tid_x]);

}
	
/*********************************************/
/* LEGENDRE POLYNOMIALS CALCULATION FUNCTION */
/*********************************************/
#define BLOCKSIZE_LEGENDRE_X	16
#define BLOCKSIZE_LEGENDRE_Y	16

template <class T>
T * generateLegendre(const T * __restrict__ d_x, const int maxDegree, const int N) {

	T *d_Leg;	gpuErrchk(cudaMalloc(&d_Leg, maxDegree * N * sizeof(T)));
	
	dim3 GridSize(iDivUp(N, BLOCKSIZE_LEGENDRE_X), iDivUp(maxDegree, BLOCKSIZE_LEGENDRE_Y));
	dim3 BlockSize(BLOCKSIZE_LEGENDRE_X, BLOCKSIZE_LEGENDRE_Y);
	generateLegendreKernel<<<GridSize, BlockSize>>>(d_Leg, d_x, maxDegree, N);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	return d_Leg;

}

template float  *  generateLegendre<float> (const float  * __restrict__, const int, const int);
template double *  generateLegendre<double>(const double * __restrict__, const int, const int);
