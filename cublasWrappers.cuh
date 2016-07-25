#ifndef CUBLASWRAPPERS_CUH
#define CUBLASWRAPPERS_CUH

/**************/
/* CUBLASTDOT */
/**************/
cublasStatus_t cublasTdot(cublasHandle_t, int, const float  *, int, const float  *, int, float  *);
cublasStatus_t cublasTdot(cublasHandle_t, int, const double *, int, const double *, int, double *);

/***************/
/* CUBLASTAXPY */
/***************/
cublasStatus_t cublasTaxpy(cublasHandle_t, int, const float  *, const float  *, int, float  *, int);
cublasStatus_t cublasTaxpy(cublasHandle_t, int, const double *, const double *, int, double *, int);

#endif
