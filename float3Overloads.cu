#include "float3Overloads.cuh"

// --- Sum between two real three dimensional vectors
__host__ __device__ float3 operator+(const float3 &a, const float3 &b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }

// --- Difference between two real three dimensional vectors
__host__ __device__ float3 operator-(const float3 &a, const float3 &b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }

// --- Identity
__host__ __device__ float3 operator+(const float3 &a) { return a; }

// --- Opposite of a real three dimensional vector
__host__ __device__ float3 operator-(const float3 &a) { return make_float3(-a.x, -a.y, -a.z); }

// --- Elementwise multiplication between two real three dimensional vectors
__host__ __device__ float3 operator*(const float3 &a, const float3 &b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }

// --- Multiplication between a real scalar and a real three dimensional vector
__host__ __device__ float3 operator*(const float a, const float3 &b) { return make_float3(a * b.x, a * b.y, a * b.z); }

// --- Multiplication between a real three dimensional vector and a scalar
__host__ __device__ float3 operator*(const float3 &a, const float b) { return make_float3(a.x * b, a.y * b, a.z * b); }

// --- Elementwise division between two real three dimensional vectors
__host__ __device__ float3 operator/(const float3 &a, const float3 &b) { return make_float3(a.x / b.x, a.y / b.y, a.z / b.z); }

// --- Scalar product between two real three dimensional vectors
__host__ __device__ float dot(const float3 &a, const float3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// --- Vector product between two real three dimensional vectors
__host__ __device__ float3 cross(const float3 &a, const float3 &b) { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }

// --- Normalize a real three dimensional vector
__host__ __device__ float3 normalize(const float3 &a) { return a * rsqrtf(dot(a, a)); }

// --- Norm of a real three dimensional vector
__host__ __device__ float norm(const float3 &a) { return sqrtf(dot(a, a)); }

// --- Distance between two real three dimensional vectors
__host__ __device__ float dist(const float3 &a, const float3 &b) { return norm(b - a); }