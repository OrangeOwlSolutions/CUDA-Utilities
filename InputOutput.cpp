#include <fstream>
#include <iomanip>
#include <stdlib.h>

#include "InputOutput.h"

#define prec_save 15

/***********************************************/
/* SAVE INDIVIDUAL REAL CPU MATRIX TO txt FILE */
/***********************************************/
template <class T>
void saveCPUrealtxt(const T * h_in, const char *filename, const int M) {

	std::ofstream outfile;
	outfile.open(filename);
	for (int i = 0; i < M; i++) outfile << std::setprecision(prec_save) << h_in[i] << "\n";
	outfile.close();

}

template void saveCPUrealtxt<float>(const float  *, const char *, const int);
template void saveCPUrealtxt<double>(const double *, const char *, const int);

/**************************************************/
/* SAVE INDIVIDUAL INTEGER CPU MATRIX TO txt FILE */
/**************************************************/
template <class T>
void saveCPUintegertxt(const T * h_in, const char *filename, const int M) {

	std::ofstream outfile;
	outfile.open(filename);
	for (int i = 0; i < M; i++) outfile << h_in[i] << "\n";
	outfile.close();

}

template void saveCPUintegertxt<int>(const int  *, const char *, const int);

/****************************************************/
/* LOAD INDIVIDUAL REAL MATRIX FROM txt FILE TO CPU */
/****************************************************/
// --- Load individual real matrix from txt file
template <class T>
void loadCPUrealtxt(T * __restrict h_out, const char *filename, const int M) {
//T * loadCPUrealtxt(T * __restrict h_out, const char *filename, const int M) {

	//h_out = (T *)malloc(M * sizeof(T));

	std::ifstream infile;
	infile.open(filename);
	for (int i = 0; i < M; i++) {
		double temp;
		infile >> temp;
		h_out[i] = (T)temp;
	}

	infile.close();

	//return h_out;

}

template void loadCPUrealtxt<int>(int * __restrict, const char *, const int);
template void loadCPUrealtxt<float>(float  * __restrict, const char *, const int);
template void loadCPUrealtxt<double>(double * __restrict, const char *, const int);
//template float  * loadCPUrealtxt<float>(float  * __restrict, const char *, const int);
//template double * loadCPUrealtxt<double>(double * __restrict, const char *, const int);


