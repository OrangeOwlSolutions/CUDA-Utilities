#ifndef INPUTOUTPUT_H
#define INPUTOUTPUT_H

template <class T>
void saveCPUrealtxt(const T *, const char *, const int);

template <class T>
T * loadCPUrealtxt(const char *, T * __restrict__, const int);

#endif
