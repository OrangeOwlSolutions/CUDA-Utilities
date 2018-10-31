#include "cublasWrappers.cuh"

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
/* CUBLASTNRM2 */
/***************/
cublasStatus_t cublasTnrm2(cublasHandle_t handle, int n, const float *x, int incx, float *result) {
	return cublasSnrm2(handle, n, x, incx, result);
}

cublasStatus_t cublasTnrm2(cublasHandle_t handle, int n, const double *x, int incx, double *result) {
	return cublasDnrm2(handle, n, x, incx, result);
}

/***************/
/* CUBLASTSCAL */
/***************/
cublasStatus_t cublasTscal(cublasHandle_t handle, int n, const float *alpha, float *x, int incx) {
	return cublasSscal(handle, n, alpha, x, incx);
}
	
cublasStatus_t cublasTscal(cublasHandle_t handle, int n, const double *alpha, double *x, int incx) {
	return cublasDscal(handle, n, alpha, x, incx);
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

/***************/
/* CUBLASTCOPY */
/***************/
cublasStatus_t cublasTcopy(cublasHandle_t handle, int n, const float *x, int incx, float *y, int incy) {
	return cublasScopy(handle, n, x, incx, y, incy);
}

cublasStatus_t cublasTcopy(cublasHandle_t handle, int n, const double *x, int incx, double *y, int incy) {
	return cublasDcopy(handle, n, x, incx, y, incy);
}

 
