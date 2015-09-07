#include <thrust/device_vector.h>

#include "Utilities.cuh"

/************/
/* LINSPACE */
/************/
template <class T>
T * linspace(const T a, const T b, const unsigned int N) {
	
	T *out_array; gpuErrchk(cudaMalloc((void**)&out_array, N * sizeof(T)));

	T Dx = (b-a)/(T)(N-1);
   
	thrust::device_ptr<T> d = thrust::device_pointer_cast(out_array); 	
	thrust::transform(thrust::make_counting_iterator(a/Dx), thrust::make_counting_iterator((b+1.f)/Dx), thrust::make_constant_iterator(Dx), d, thrust::multiplies<T>());

	return out_array;
}

template float  * linspace<float> (const float  a, const float  b, const unsigned int N);
template double * linspace<double>(const double a, const double b, const unsigned int N);
