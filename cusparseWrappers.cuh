#ifndef CUSPARSEWRAPPERS_CUH
#define CUSPARSEWRAPPERS_CUH

cusparseStatus_t cusparseTcsrmv(cusparseHandle_t, cusparseOperation_t, int, int, int, const float *, const cusparseMatDescr_t, const float *, 
	const int *, const int *, const float *, const float *, float *);

cusparseStatus_t cusparseTcsrmv(cusparseHandle_t, cusparseOperation_t, int, int, int, const double *, const cusparseMatDescr_t, const double *, 
	const int *, const int *, const double *, const double *, double *);

#endif
