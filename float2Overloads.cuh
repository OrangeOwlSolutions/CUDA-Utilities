#ifndef __FLOAT2OVERLOADS_CUH__
#define __FLOAT2OVERLOADS_CUH__

#include <cuda.h>

// --- Real part of a complex number
__host__ __device__ const float &REAL(const float2 &a);

// --- Imaginary part of a complex number
__host__ __device__ const float &IMAG(const float2 &a);

// --- Sum between two complex numbers
__host__ __device__ float2 operator+(const float2 &a, const float2 &b);

// --- Difference between two complex numbers
__host__ __device__ float2 operator-(const float2 &a, const float2 &b);

// --- Opposite of a complex number
__host__ __device__ float2 operator-(const float2 &a);

// --- Identity
__host__ __device__ float2 operator+(const float2 &a);

// --- Multiplication between two complex numbers
__host__ __device__ float2 operator*(const float2 &a, const float2 &b);

// --- Multiplication between a real number and a complex number
__host__ __device__ float2 operator*(const float &a, const float2 &b);

// --- Multiplication between a complex number and a real number
__host__ __device__ float2 operator*(const float2 &a, const float &b);

// --- Division between a complex number and a real number
__host__ __device__ float2 operator/(const float2 &a, const float &b);

// --- Division between a real number and a complex number
__host__ __device__ float2 operator/(const float &a, const float2 &b);

// --- Division between two complex numbers
__host__ __device__ float2 operator/(const float2 &a, const float2 &b);

// --- Conjugate of a complex number
__host__ __device__ float2 conjugate(const float2 &a);

// --- Exponential of a complex number
__host__ __device__ float2 expf(const float2 &a);

// --- Square root of a complex number
__host__ __device__ float2 sqrtf(const float2 &z);

// --- Absolute value of a complex number
__host__ __device__ float fabsf(const float2 &a);

#endif


