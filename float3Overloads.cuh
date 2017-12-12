#ifndef __FLOAT3OVERLOADS_CUH__
#define __FLOAT3OVERLOADS_CUH__

#include <cuda.h>

// --- Sum between two real three dimensional vectors
__host__ __device__ float3 operator+(const float3 &a, const float3 &b);

// --- Difference between two real three dimensional vectors
__host__ __device__ float3 operator-(const float3 &a, const float3 &b);

// --- Identity
__host__ __device__ float3 operator+(const float3 &a);

// --- Opposite of a real three dimensional vector
__host__ __device__ float3 operator-(const float3 &a);

// --- Elementwise multiplication between two real three dimensional vectors
__host__ __device__ float3 operator*(const float3 &a, const float3 &b);

// --- Multiplication between a real scalar and a real three dimensional vector
__host__ __device__ float3 operator*(const float a, const float3 &b);

// --- Multiplication between a real three dimensional vector and a scalar
__host__ __device__ float3 operator*(const float3 &a, const float b);

// --- Elementwise division between two real three dimensional vectors
__host__ __device__ float3 operator/(const float3 &a, const float3 &b);

// --- Scalar product between two real three dimensional vectors
__host__ __device__ float dot(const float3 &a, const float3 &b);

// --- Vector product between two real three dimensional vectors
__host__ __device__ float3 cross(const float3 &a, const float3 &b);

// --- Normalize a real three dimensional vector
__host__ __device__ float3 normalize(const float3 &a);

// --- Norm of a real three dimensional vector
__host__ __device__ float norm(const float3 &a);

// --- Distance between two real three dimensional vectors
__host__ __device__ float dist(const float3 &a, const float3 &b);

#endif
