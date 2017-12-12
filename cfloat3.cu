#include "float2Overloads.cuh"
#include "cfloat3.cuh"

// --- Make a complex three dimensional vector from its complex components
__host__ __device__ cfloat3 make_cfloat3(const float2 &x, const float2 & y, const float2 &z) { cfloat3 cv; cv.x = x; cv.y = y; cv.z = z; return cv; }

// --- Sum between two complex three dimensional vectors 
__host__ __device__ cfloat3 operator+(const cfloat3 &a, const cfloat3 &b) { return make_cfloat3(a.x + b.x, a.y + b.y, a.z + b.z); }

// --- Difference between two complex three dimensional vectors 
__host__ __device__ cfloat3 operator-(const cfloat3 &a, const cfloat3 &b) { return make_cfloat3(a.x - b.x, a.y - b.y, a.z - b.z); }

// --- Identity
__host__ __device__ cfloat3 operator+(const cfloat3 &a) { return a; }

// --- Opposite of a complex three dimensional vector
__host__ __device__ cfloat3 operator-(const cfloat3 &a) { return make_cfloat3(-a.x, -a.y, -a.z); }

// --- Elementwise multiplication between two complex three dimensional vectors
__host__ __device__ cfloat3 operator*(const cfloat3 &a, const cfloat3 &b) { return make_cfloat3(a.x * b.x, a.y * b.y, a.z * b.z); }

// --- Multiplication between a complex scalar and a complex three dimensional vector
__host__ __device__ cfloat3 operator*(const float2 &a, const cfloat3 &b) { return make_cfloat3(a * b.x, a * b.y, a * b.z); }

// --- Multiplication between a complex three dimensional vector and a complex scalar
__host__ __device__ cfloat3 operator*(const cfloat3 &a, const float2 &b) { return make_cfloat3(a.x * b, a.y * b, a.z * b); }

// --- Multiplication between a real scalar and a complex three dimensional vector
__host__ __device__ cfloat3 operator*(const float a, const cfloat3 &b) { return make_cfloat3(a * b.x, a * b.y, a * b.z); }

// --- Multiplication between a complex three dimensional vector and a real scalar
__host__ __device__ cfloat3 operator*(const cfloat3 &a, const float b) { return make_cfloat3(a.x * b, a.y * b, a.z * b); }

// --- Complex, non-hermitian scalar product between two complex three dimensional vectors
__host__ __device__ float2 edot(const cfloat3 &a, const cfloat3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// --- Complex, non-hermitian scalar product between a complex three dimensional vector and a real three dimensional vector
__host__ __device__ float2 edot(const cfloat3 &a, const float3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// --- Complex, non-hermitian scalar product between a real three dimensional vector and a complex three dimensional vector 
__host__ __device__ float2 edot(const float3 &a, const cfloat3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// --- Vector product between a real three dimensional vector and a complex three dimensional vector
__host__ __device__ cfloat3 cross(const float3 &a, const cfloat3 &b) { return make_cfloat3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }

// --- Vector product between a complex three dimensional vector and a real three dimensional vector
__host__ __device__ cfloat3 cross(const cfloat3 &a, const float3 &b) { return make_cfloat3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }

// --- Conjugate of a complex three dimensional vector
__host__ __device__ cfloat3 conjugate(const cfloat3 &a) { return make_cfloat3(conjugate(a.x), conjugate(a.y), conjugate(a.z)); }

// --- Complex scalar product between two complex three dimensional vectors
__host__ __device__ float2 dot(const cfloat3 &a, const cfloat3 &b) { return edot(a, conjugate(b)); }

// --- Norm of a complex three dimensional vector
__host__ __device__ float  norm(const cfloat3 &a) {
	return sqrtf(REAL(a.x) * REAL(a.x) + REAL(a.y) * REAL(a.y) + REAL(a.z) * REAL(a.z) +
		IMAG(a.x) * IMAG(a.x) + IMAG(a.y) * IMAG(a.y) + IMAG(a.z) * IMAG(a.z));
}

// --- Multiplication between a real three dimensional vector and a complex scalar
__host__ __device__ cfloat3 operator*(const float3 &a, const float2 &b) { return make_cfloat3(a.x*b, a.y*b, a.z*b); }

// --- Multiplication between a complex scalar and a real three dimensional vector
__host__ __device__ cfloat3 operator*(const float2 &a, const float3 &b) { return make_cfloat3(a * b.x, a * b.y, a * b.z); }

// --- Sum between a real three dimensional vector and a complex three dimensional vector
__host__ __device__ cfloat3 operator+(const float3 &a, const cfloat3 &b) { return make_cfloat3(make_float2(a.x, 0.f) + b.x, make_float2(a.y, 0.f) + b.y, make_float2(a.z, 0.f) + b.z); }

// --- Sum between a complex three dimensional vector and real three dimensional vector
__host__ __device__ cfloat3 operator+(const cfloat3 &a, const float3 &b) { return make_cfloat3(make_float2(b.x, 0.f) + a.x, make_float2(b.y, 0.f) + a.y, make_float2(b.z, 0.f) + a.z); }

// --- Division between a complex three dimensional vector and a real scalar
__host__ __device__ cfloat3 operator/(const cfloat3 &a, const float b) { return make_cfloat3(a.x / b, a.y / b, a.z / b); }

// --- Division between a complex three dimensional vector and a complex scalar
__host__ __device__ cfloat3 operator/(const cfloat3 &a, const float2 &b) { return make_cfloat3(a.x / b, a.y / b, a.z / b); }


