#ifndef POLYNOMIALS_CUH
#define POLYNOMIALS_CUH

#include <cuda_runtime.h>
#include <thrust/pair.h>

/************/
/* LEGENDRE */
/************/
template <class T> 
__host__ __device__ T LegendreN(unsigned int, const T&);

template <class T>
T * generateLegendre(const T * __restrict__, const int, const int);

template <class T>
thrust::pair<thrust::pair<T *, T*>, T*> generateLegendreFactorized(const int, const int, const int, const int);

template <class T>
thrust::pair<thrust::pair<T *, T*>, T*> h_generateLegendreFactorized(const int, const int, const int, const int);

/***********/
/* ZERNIKE */
/***********/
__host__ __device__ unsigned int binomial_coefficient(unsigned int n, unsigned int k);

enum ZernikeEvaluationMethod {direct, recursion};

template <class T> __host__ __device__ T ZernikeRnm(const unsigned int n, const unsigned int m, const T r, ZernikeEvaluationMethod eval_method = recursion);

template <class T> __host__ __device__ T ZernikeRnmLow(const unsigned int n, const unsigned int m, T r);

template <class T> __host__ __device__ T ZernikeRnmDirect(const unsigned int n, const unsigned int m, const T r);

template <class T> __host__ __device__ T ZernikeRnmRecursion(const unsigned int n, const unsigned int m, const T r);

template <class T> __host__ __device__ T Zernikenm(const unsigned int n, const int m, const T r, const T theta, const bool normalize = 0, const ZernikeEvaluationMethod eval_method = direct);

template <class T> __host__ __device__ T Zernikep(const unsigned int p, const T r, const T theta, const bool normalize = 0, const ZernikeEvaluationMethod eval_method = direct);

template <class T> T * generateZernikep(const T * __restrict__, const T * __restrict__, const int, const int, const int);

template <class T> T * h_generateZernikep(const T * __restrict__, const T * __restrict__, const int, const int, const int);

#endif
