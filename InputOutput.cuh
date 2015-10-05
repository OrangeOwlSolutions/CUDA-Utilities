#ifndef INPUTOUTPUT_H
#define INPUTOUTPUT_H

template <class T>
void saveGPUrealtxt(const T *, const char *, const int);

template <class T>
void saveCPUrealtxt(const T *, const char *, const int);

template <class T>
void saveGPUcomplextxt(const T *, const char *, const int);

void saveGPUcomplextxt(const double2 *, const char *, const int);

template <class T>
T * loadGPUrealtxt(const char *, T * __restrict__, const int);

#endif
