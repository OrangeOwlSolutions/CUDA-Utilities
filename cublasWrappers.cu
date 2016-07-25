#include "cublas_v2.h"

/**************/
/* CUBLASTDOT */
/**************/
cublasStatus_t cublasTdot(cublasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result) {
	return cublasSdot(handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublasTdot(cublasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result) {
	return cublasDdot(handle, n, x, incx, y, incy, result);
}

/***************/
/* CUBLASTAXPY */
/***************/
cublasStatus_t cublasTaxpy(cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy) {
	return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublasTaxpy(cublasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy) {
	return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

