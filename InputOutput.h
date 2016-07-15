#ifndef INPUTOUTPUT_H
#define INPUTOUTPUT_H

template <class T>
void saveCPUrealtxt(const T *, const char *, const int);

template <class T>
void saveCPUintegertxt(const T *, const char *, const int);

template <class T>
void loadCPUrealtxt(T * __restrict, const char *, const int);
//T * loadCPUrealtxt(T * __restrict, const char *, const int);

#endif
