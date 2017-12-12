#ifndef MATLAB_LIKE_CUH
#define MATLAB_LIKE_CUH

/************/
/* LINSPACE */
/************/
template <class T> T * d_linspace(const T, const T, const unsigned int);

template <class T> T * h_linspace(const T, const T, const unsigned int);

/*********/
/* ZEROS */
/*********/
template <class T> T * h_zeros(const int M, const int N);

/************/
/* MESHGRID */
/************/
#include <thrust/pair.h>

template <class T>
thrust::pair<T *, T *> d_meshgrid(const T *, const unsigned int, const T *, const unsigned int);

template <class T>
thrust::pair<T *, T *> h_meshgrid(const T *, const unsigned int, const T *, const unsigned int);

/*********/
/* COLON */
/*********/
template <class T>
T * d_colon(const T, const T, const T);

template <class T>
T * h_colon(const T, const T, const T);

/*****************************/
/* GENERATE SYMMETRIC POINTS */
/*****************************/
template<class T>
T * generateSymmetricPoints(const T step, const T b);

#endif
