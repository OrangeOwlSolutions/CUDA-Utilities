#include "Utilities.cuh"
#include "Polynomials.cuh"
#include "Matlab_like.cuh"

#include <thrust/pair.h>

#define DEBUG

#define pi 3.141592653589793238463

// --- The approach for the Legendre polynomials could be improved by calculating the coefficients of x^n and then summing up the polynomials using fused multiply add functions
// --- The approach for the Zernike polynomials could be improved by constructing proper pow functions with integer exponent

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
template <class T> __host__ __device__ T LegendreN(unsigned int n, const T& x) {
	
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

template __host__ __device__ float   LegendreN<float> (unsigned int, const float &);
template __host__ __device__ double  LegendreN<double>(unsigned int, const double&);

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

/******************************************************/
/* FACTORIZED LEGENDRE POLYNOMIALS CALCULATION KERNEL */
/******************************************************/
template <class T>
__global__ void generateLegendreFactorizedKernel(T * __restrict__ d_Leg, const T * __restrict__ d_X, const T * __restrict__ d_Y, 
	                                             const int maxDegreeX, const int maxDegreeY, const int N) {

	const int tid_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int tid_y = blockDim.y * blockIdx.y + threadIdx.y;
	const int tid_z = blockDim.z * blockIdx.z + threadIdx.z;

	if ((tid_x < N) && (tid_y < maxDegreeX) && (tid_z < maxDegreeY)) d_Leg[(tid_z * maxDegreeX + tid_y) * N + tid_x] = LegendreN(tid_z, d_X[tid_x]) * LegendreN(tid_y, d_Y[tid_x]);

}
	
/********************************************************/
/* FACTORIZED LEGENDRE POLYNOMIALS CALCULATION FUNCTION */
/********************************************************/
#define BLOCKSIZE_LEGENDRE_FACTORIZED_X		8
#define BLOCKSIZE_LEGENDRE_FACTORIZED_Y		4
#define BLOCKSIZE_LEGENDRE_FACTORIZED_Z		4

template <class T>
thrust::pair<thrust::pair<T *, T*>, T*> generateLegendreFactorized(const int maxDegreeX, const int maxDegreeY, const int M_x, const int M_y) {

	// --- Generating the (csi, eta) grid
	T *d_csi = linspace(-static_cast<T>(1), static_cast<T>(1), M_x);
	T *d_eta = linspace(-static_cast<T>(1), static_cast<T>(1), M_y);

	thrust::pair<T *, T *> d_CSI_ETA = meshgrid(d_csi, M_x, d_eta, M_y);
	T *d_CSI = d_CSI_ETA.first;
	T *d_ETA = d_CSI_ETA.second;

	// --- Generating the Legendre polynomials
	const int N = M_x * M_y;
	
	T *d_Leg;	gpuErrchk(cudaMalloc(&d_Leg, maxDegreeX * maxDegreeY * N * sizeof(T)));
	
	dim3 GridSize(iDivUp(N, BLOCKSIZE_LEGENDRE_FACTORIZED_X), iDivUp(maxDegreeX, BLOCKSIZE_LEGENDRE_FACTORIZED_Y), iDivUp(maxDegreeY, BLOCKSIZE_LEGENDRE_FACTORIZED_Z));
	dim3 BlockSize(BLOCKSIZE_LEGENDRE_FACTORIZED_X, BLOCKSIZE_LEGENDRE_FACTORIZED_Y, BLOCKSIZE_LEGENDRE_FACTORIZED_Z);
	generateLegendreFactorizedKernel<<<GridSize, BlockSize>>>(d_Leg, d_CSI, d_ETA, maxDegreeX, maxDegreeY, N);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	gpuErrchk(cudaFree(d_csi));
	gpuErrchk(cudaFree(d_eta));

	return thrust::make_pair(thrust::make_pair(d_CSI, d_ETA), d_Leg);

}

template thrust::pair<thrust::pair<float  *, float *>, float *>  generateLegendreFactorized<float> (const int, const int, const int, const int);
template thrust::pair<thrust::pair<double *, double*>, double*>  generateLegendreFactorized<double>(const int, const int, const int, const int);

/************************/
/* BINOMIAL COEFFICIENT */
/************************/
__host__ __device__ unsigned int binomial_coefficient(const unsigned int n, const unsigned int k)
{
	double binom = static_cast<double>(1);

	for (unsigned int i = 1; i <= k; i++) binom *= static_cast<double>(n-(k-i))/static_cast<double>(i);

	return static_cast<unsigned int>(binom);
}

/***********************************************/
/* ZERNIKE RADIAL POLYNOMIALS - LOW ORDER CASE */
/***********************************************/
// --- See "Radial Polynomials" at https://en.wikipedia.org/wiki/Zernike_polynomials
template<class T>
__host__ __device__ T ZernikeRnmLow(const unsigned int n, const unsigned int m, const T r) {
	
	// --- Broken function if order n > 6
	if (n > 6) { return static_cast<T>(0); }

	if (n == 2 && m == 0) return static_cast<T>(2) * std::pow(r, 2) - static_cast<T>(1);

	if (n == 3 && m == 1) return static_cast<T>(3) * std::pow(r, 3) - static_cast<T>(2) * r;

	if (n == 4) {
		
		if (m == 0) return static_cast<T>(6) * std::pow(r, 4) - static_cast<T>(6) * std::pow(r, 2) + static_cast<T>(1);

		if (m == 2) return static_cast<T>(4) * std::pow(r, 4) - static_cast<T>(3) * std::pow(r, 2);
	}

	if (n == 5) {
    
		if (m == 1) return static_cast<T>(10) * std::pow(r, 5) - static_cast<T>(12) * std::pow(r, 3) + static_cast<T>(3) * r;

		if (m == 3) return static_cast<T>(5) *  std::pow(r, 5) - static_cast<T>(4) * std::pow(r, 3);
	} 

	if (n == 6) {
    
		if (m == 0) return static_cast<T>(20) * std::pow(r, 6) - static_cast<T>(30) * std::pow(r, 4) + static_cast<T>(12) * std::pow(r, 2) - static_cast<T>(1);

		if (m == 2) return static_cast<T>(15) * std::pow(r, 6) - static_cast<T>(20) * std::pow(r, 4) + static_cast<T>(6) * std::pow(r, 2);

		if (m == 4) return static_cast<T>(6)  * std::pow(r, 6) - static_cast<T>(5)  * std::pow(r, 4);
	}
}

/***************************************************************/
/* ZERNIKE RADIAL POLYNOMIALS - GENERAL CASE - DIRECT APPROACH */
/***************************************************************/
// --- See Other Representations at https://en.wikipedia.org/wiki/Zernike_polynomials
template<class T>
__host__ __device__ T ZernikeRnmDirect(const unsigned int n, const unsigned int m, const T r)
{	
	T r_pow					= pow(r, static_cast<T>(m));
	unsigned int npm2		= (n+m)/2;
	unsigned int nmm2		= (n-m)/2;
	T a_k					= pow(static_cast<T>(-1), static_cast<T>(nmm2)) *binomial_coefficient(npm2, nmm2);

	T zernike_value  = a_k * r_pow;

	for (int k = nmm2 - 1; k >= 0; k--) {
		a_k				*= -static_cast<T>((k + 1) * (n - k)) / static_cast<T>((npm2 - k) * (nmm2 - k));
		r_pow			*= pow(r, static_cast<T>(2));
		zernike_value	+= a_k * r_pow;
	}

	return zernike_value;
}

/******************************************************************/
/* ZERNIKE RADIAL POLYNOMIALS - GENERAL CASE - RECURSION APPROACH */
/******************************************************************/
// --- See http://repository.um.edu.my/31715/1/ol-38-14-2487.pdf
template<class T>
__host__ __device__ T ZernikeRnmRecursion(const unsigned int n, const unsigned int m, const T r)
{
	// --- Number of diagonals that need to be computed
	unsigned int ndiags    = (n-m)/2;

	// --- Last element for which n = m
	unsigned int nm        = n-(n-m)/2;
	unsigned int diag_size = nm+1;

	// --- Initial diagonal
	T *zernike_values_old = (T *)malloc(diag_size * sizeof(T));
	T *zernike_values_new = (T *)malloc(diag_size * sizeof(T));
	memset(zernike_values_old, static_cast<T>(0), diag_size * sizeof(T));
	memset(zernike_values_new, static_cast<T>(0), diag_size * sizeof(T));

	for (unsigned int i=0; i < diag_size; i++) zernike_values_old[i] = pow(r, static_cast<T>(i));
  
	--diag_size;

	for (unsigned int i = 0; i < ndiags; i++) {
		for (unsigned int j = 0; j < diag_size; j++) {

			if (j==0)	zernike_values_new[0] = -zernike_values_old[0] + static_cast<T>(2) * r * zernike_values_old[1];

			else		zernike_values_new[j] = -zernike_values_old[j] + r * (zernike_values_new[j - 1] + zernike_values_old[j + 1]);
		}
		zernike_values_old = zernike_values_new;
		--diag_size;
	}

	return zernike_values_new[diag_size];
}

/******************************/
/* ZERNIKE RADIAL POLYNOMIALS */
/******************************/
template<class T>
__host__ __device__ T ZernikeRnm(const unsigned int n, const unsigned int m, const T r, const ZernikeEvaluationMethod eval_method) {
	
	// --- the Zernike polynomials are undefined if n<m or n-m is not even
	if (n < m) { return static_cast<T>(0); }

	// --- Evaluates the Zernike polynomials on the diagonal n=m.
	if (n == m) { return std::pow(r, static_cast<T>(n)); }

	// --- If the order is low enough, "hard-coded" polynomials are used.
	if (n <= 6) return ZernikeRnmLow(n, m, r);

	// --- Otherwise, we use the recursion relation, or the direct approach, as chosen by the user.
	T zernike_value;
  
	switch (eval_method) {
		
		case direct:		{ zernike_value = ZernikeRnmDirect(n, m, r);		break; }

		case recursion:		{ zernike_value = ZernikeRnmRecursion(n, m, r);		break; }

		default:			{ zernike_value = ZernikeRnmRecursion(n, m, r);    break; }
  }

	return zernike_value;
}

/**************************/
/* ZERNIKE POLYNOMIALS nm */
/**************************/
template <class T>
__host__ __device__ T Zernikenm(const unsigned int n, const int m, const T r, const T theta, const bool normalize, const ZernikeEvaluationMethod eval_method)
{
	const unsigned int m_abs	= abs(m);
	const T normalization		= sqrt((static_cast<T>(2)*n+static_cast<T>(2)) / 
		                              ((static_cast<T>(1)+(m == 0 ? static_cast<T>(1) : static_cast<T>(0))) * static_cast<T>(pi)));
	const T radial_poly			= ZernikeRnm(n, m_abs, r, eval_method);
	const T angular_poly		= (m > 0 ? sin(m_abs * theta) : cos(m_abs * theta));

    if (normalize) return normalization * radial_poly * angular_poly;
	else           return				  radial_poly * angular_poly;
}

/*************************/
/* ZERNIKE POLYNOMIALS p */
/*************************/
template <class T>
__host__ __device__ T Zernikep(const unsigned int p, const T r, const T theta, const bool normalize, const ZernikeEvaluationMethod eval_method)
{
	const unsigned int n					= ceil((-3. + sqrt(9. + 8. * p)) / 2.);
	const		   int m					= 2 * p - n * (n + 2);

	return Zernikenm(n, m, r, theta, normalize, eval_method);
}

template __host__ __device__ float  Zernikep<float> (const unsigned int p, const float  r, const float  theta, const bool normalize, const ZernikeEvaluationMethod eval_method);
template __host__ __device__ double Zernikep<double>(const unsigned int p, const double r, const double theta, const bool normalize, const ZernikeEvaluationMethod eval_method);

/******************************************/
/* ZERNIKE POLYNOMIALS CALCULATION KERNEL */
/******************************************/
#define BLOCKSIZE_ZERNIKE_X	16
#define BLOCKSIZE_ZERNIKE_Y	16

template <class T>
__global__ void generateZernikepKernel(T * __restrict__ d_Zernike, const T * __restrict__ d_rho, const T * __restrict__ d_theta, const int maxDegree, const int N) {

	const int tid_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int tid_y = blockDim.y * blockIdx.y + threadIdx.y;

	if ((tid_x < N) && (tid_y < maxDegree)) d_Zernike[tid_y * N + tid_x] = Zernikep(tid_y + 1, d_rho[tid_x], d_theta[tid_x]);

}
	
/**************************************************************************/
/* ZERNIKE POLYNOMIALS CALCULATION FUNCTION - SINGLE INDEX - (x, y) INPUT */
/**************************************************************************/

// --- Zernike polynomials with single index.
template <class T>
T * generateZernikep(const T * __restrict__ d_CSI, const T * __restrict__ d_ETA, const int maxDegree, const int M_x, const int M_y) {

	thrust::pair<T *, T *> d_RHO_THETA = Cartesian2Polar(d_CSI, d_ETA, M_x * M_y, static_cast<T>(1) / sqrt(static_cast<T>(2)));
	T *d_RHO	= d_RHO_THETA.first;
	T *d_THETA  = d_RHO_THETA.second;

	const int N = M_x * M_y;
	
	T *d_Zernike;	gpuErrchk(cudaMalloc(&d_Zernike, N * maxDegree * sizeof(T)));
	
	dim3 GridSize(iDivUp(N, BLOCKSIZE_ZERNIKE_X), iDivUp(maxDegree, BLOCKSIZE_ZERNIKE_Y));
	dim3 BlockSize(BLOCKSIZE_ZERNIKE_X, BLOCKSIZE_ZERNIKE_Y);
	generateZernikepKernel<<<GridSize, BlockSize>>>(d_Zernike, d_RHO, d_THETA, maxDegree, N);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	return d_Zernike;
}

template float  * generateZernikep<float> (const float  * __restrict__, const float  * __restrict__, const int, const int, const int);
template double * generateZernikep<double>(const double * __restrict__, const double * __restrict__, const int, const int, const int);
