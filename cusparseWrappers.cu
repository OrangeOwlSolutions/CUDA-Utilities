#include "cusparse_v2.h"

/******************/
/* CUSPARSETCSRMV */
/******************/
cusparseStatus_t cusparseTcsrmv(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, const float *alpha,
	const cusparseMatDescr_t descrA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, const float *x, const float *beta, float *y)  {
	return cusparseScsrmv(handle, transA, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y);
}

cusparseStatus_t cusparseTcsrmv(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, const double *alpha,
	const cusparseMatDescr_t descrA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, const double *x, const double *beta, 
	double *y) {
	return cusparseDcsrmv(handle, transA, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y); }
