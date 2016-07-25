#ifndef PRECONDCONJUGATEGRADIENTSPARSE_CUH
#define PRECONDCONJUGATEGRADIENTSPARSE_CUH

template<class T>
void precondConjugateGradientSparse(const int * __restrict__, const int, const int * __restrict__, const T * __restrict__, const int,
	T * __restrict__, const int, T * __restrict__, int &);

#endif
