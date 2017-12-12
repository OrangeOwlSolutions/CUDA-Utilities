#include "float2Overloads.cuh"

// --- Real part of a complex number
__host__ __device__ const float &REAL(const float2 &a) { return a.x; }

// --- Imaginary part of a complex number
__host__ __device__ const float &IMAG(const float2 &a) { return a.y; }

// --- Sum between two complex numbers
__host__ __device__ float2 operator+(const float2 &a, const float2 &b) { return make_float2(REAL(a) + REAL(b), IMAG(a) + IMAG(b)); }

// --- Difference between two complex numbers
__host__ __device__ float2 operator-(const float2 &a, const float2 &b) { return make_float2(REAL(a) - REAL(b), IMAG(a) - IMAG(b)); }

// --- Opposite of a complex number
__host__ __device__ float2 operator-(const float2 &a) { return make_float2(-REAL(a), -IMAG(a)); }

// --- Identity
__host__ __device__ float2 operator+(const float2 &a) { return a; }

// --- Multiplication between two complex numbers
__host__ __device__ float2 operator*(const float2 &a, const float2 &b) { return make_float2(REAL(a) * REAL(b) - IMAG(a) * IMAG(b), REAL(a) * IMAG(b) + IMAG(a) * REAL(b)); }

// --- Multiplication between a real number and a complex number
__host__ __device__ float2 operator*(const float &a, const float2 &b) { return make_float2(a * REAL(b), a * IMAG(b)); }

// --- Multiplication between a complex number and a real number
__host__ __device__ float2 operator*(const float2 &a, const float &b) { return make_float2(b * REAL(a), b * IMAG(a)); }

// --- Division between a complex number and a real number
__host__ __device__ float2 operator/(const float2 &a, const float &b) { return make_float2(REAL(a) / b, IMAG(a) / b); }

// --- Division between a real number and a complex number
__host__ __device__ float2 operator/(const float &a, const float2 &b) {

	const float den = hypotf(REAL(b), IMAG(b));
	const float sq_den = den*den;
	const float m = a / sq_den;

	return make_float2(REAL(b) * m, -IMAG(b) * m);
}

// --- Division between two complex numbers
__host__ __device__ float2 operator/(const float2 &a, const float2 &b) {

	float2 c;
	const float den = hypot(REAL(b), IMAG(b));
	const float sq_den = den * den;
	c.x = (REAL(a) * REAL(b) + IMAG(a) * IMAG(b)) / sq_den;
	c.y = (IMAG(a) * REAL(b) - REAL(a) * IMAG(b)) / sq_den;
	return c;
}

// --- Conjugate of a complex number
__host__ __device__ float2 conjugate(const float2 &a) { return make_float2(REAL(a), -IMAG(a)); }

// --- Exponential of a complex number
__host__ __device__ float2 expf(const float2 &a) { return expf(REAL(a)) * make_float2(cosf(IMAG(a)), sinf(IMAG(a))); }

// --- Square root of a complex number
__host__ __device__ float2 sqrtf(const float2 &z) {

	const float rr = REAL(z);
	const float ii = IMAG(z);
	const float zero = 0.0f;
	const float half = 0.5f;

	float x = fabsf(rr);
	float y = fabsf(ii);

	if ((x == zero) & (y == zero)) return make_float2(zero, zero);

	float temp = hypotf(x, y) + x;
	x = sqrtf(half * temp);
	y = sqrtf(half * y * y / temp);
	if (ii >= zero) {
		if (rr >= 0.f)
			return make_float2(x, y);
		else
			return make_float2(y, x);
	}
	else {
		if (rr >= 0.f)
			return make_float2(x, -y);
		else
			return make_float2(y, -x);
	}
}

// --- Absolute value of a complex number
__host__ __device__ float fabsf(const float2 &a) { return hypotf(REAL(a), IMAG(a)); }

