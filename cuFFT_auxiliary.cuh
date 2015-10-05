#ifndef CUFFT_AUXILIARY_CUH
#define CUFFT_AUXILIARY_CUH

__global__ void fftshift_2D(float2  * __restrict__, const int, const int);
__global__ void fftshift_2D(double2 * __restrict__, const int, const int);

#endif
