#ifndef __CFLOAT2_CUH__
#define __CFLOAT2_CUH__

#include <cuda.h>

// --- This structure isn't 128-bit aligned
struct cfloat3 { float2 x; float2 y; float2 z; };

// --- Make a complex three dimensional vector from its complex components
__host__ __device__ cfloat3 make_cfloat3(const float2 &x, const float2 & y, const float2 &z);

// --- Sum between two complex three dimensional vectors 
__host__ __device__ cfloat3 operator+(const cfloat3 &a, const cfloat3 &b);

// --- Difference between two complex three dimensional vectors 
__host__ __device__ cfloat3 operator-(const cfloat3 &a, const cfloat3 &b);

// --- Identity
__host__ __device__ cfloat3 operator+(const cfloat3 &a);

// --- Opposite of a complex three dimensional vector
__host__ __device__ cfloat3 operator-(const cfloat3 &a);

// --- Elementwise multiplication between two complex three dimensional vectors
__host__ __device__ cfloat3 operator*(const cfloat3 &a, const cfloat3 &b);

// --- Multiplication between a complex scalar and a complex three dimensional vector
__host__ __device__ cfloat3 operator*(const float2 &a, const cfloat3 &b);

// --- Multiplication between a complex three dimensional vector and a complex scalar
__host__ __device__ cfloat3 operator*(const cfloat3 &a, const float2 &b);

// --- Multiplication between a real scalar and a complex three dimensional vector
__host__ __device__ cfloat3 operator*(const float a, const cfloat3 &b);

// --- Multiplication between a complex three dimensional vector and a real scalar
__host__ __device__ cfloat3 operator*(const cfloat3 &a, const float b);

// --- Complex, non-hermitian scalar product between two complex three dimensional vectors
__host__ __device__ float2 edot(const cfloat3 &a, const cfloat3 &b);

// --- Complex, non-hermitian scalar product between a complex three dimensional vector and a real three dimensional vector
__host__ __device__ float2 edot(const cfloat3 &a, const float3 &b);

// --- Complex, non-hermitian scalar product between a real three dimensional vector and a complex three dimensional vector 
__host__ __device__ float2 edot(const float3 &a, const cfloat3 &b);

// --- Vector product between a real three dimensional vector and a complex three dimensional vector
__host__ __device__ cfloat3 cross(const float3 &a, const cfloat3 &b);

// --- Vector product between a complex three dimensional vector and a real three dimensional vector
__host__ __device__ cfloat3 cross(const cfloat3 &a, const float3 &b);

// --- Conjugate of a complex three dimensional vector
__host__ __device__ cfloat3 conjugate(const cfloat3 &a);

// --- Complex scalar product between two complex three dimensional vectors
__host__ __device__ float2 dot(const cfloat3 &a, const cfloat3 &b);

// --- Norm of a complex three dimensional vector
__host__ __device__ float  norm(const cfloat3 &a);

// --- Multiplication between a real three dimensional vector and a complex scalar
__host__ __device__ cfloat3 operator*(const float3 &a, const float2 &b);

// --- Multiplication between a complex scalar and a real three dimensional vector
__host__ __device__ cfloat3 operator*(const float2 &a, const float3 &b);

// --- Sum between a real three dimensional vector and a complex three dimensional vector
__host__ __device__ cfloat3 operator+(const float3 &a, const cfloat3 &b);

// --- Sum between a complex three dimensional vector and real three dimensional vector
__host__ __device__ cfloat3 operator+(const cfloat3 &a, const float3 &b);

// --- Division between a complex three dimensional vector and a real scalar
__host__ __device__ cfloat3 operator/(const cfloat3 &a, const float b);

// --- Division between a complex three dimensional vector and a complex scalar
__host__ __device__ cfloat3 operator/(const cfloat3 &a, const float2 &b);

#define cfloat3_zero  (make_cfloat3(make_cfloat(0.f, 0.f), make_cfloat(0.f, 0.f), make_cfloat(0.f, 0.f)))

#endif
