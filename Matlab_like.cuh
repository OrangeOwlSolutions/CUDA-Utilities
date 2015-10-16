#ifndef MATLAB_LIKE_CUH
#define MATLAB_LIKE_CUH

/************/
/* LINSPACE */
/************/
template <class T> T * linspace(const T, const T, const unsigned int);

/************/
/* MESHGRID */
/************/
#include <thrust/pair.h>

template <class T>
thrust::pair<T *,T *> meshgrid(const T *, const unsigned int, const T *, const unsigned int);

/*********/
/* COLON */
/*********/
template <class T>
T * colon(const T, const T, const T);

/*****************************/
/* GENERATE SYMMETRIC POINTS */
/*****************************/
template<class T> 
T * generateSymmetricPoints(const T step, const T b);

#endif
