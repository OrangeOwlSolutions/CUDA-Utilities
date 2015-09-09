#ifndef POLYNOMIALS_CUH
#define POLYNOMIALS_CUH

template <class T> 
inline __host__ __device__ T LegendreN(unsigned int, const T&);

template <class T>
T * generateLegendre(const T * __restrict__, const int, const int);

#endif
