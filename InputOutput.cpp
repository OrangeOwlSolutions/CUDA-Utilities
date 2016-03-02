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
	for(int i = 0; i < M; i++) outfile << std::setprecision(prec_save) << h_in[i] << "\n"; 
	outfile.close();

}

template void saveCPUrealtxt<float> (const float  *, const char *, const int);
template void saveCPUrealtxt<double>(const double *, const char *, const int);

/****************************************************/
/* LOAD INDIVIDUAL REAL MATRIX FROM txt FILE TO CPU */
/****************************************************/
// --- Load individual real matrix from txt file
template <class T>
T * loadCPUrealtxt(const char *filename, T * __restrict__ h_out, const int M) {
		
	h_out	= (T *)malloc(M * sizeof(T));

	std::ifstream infile;
	infile.open(filename);
	for(int i = 0; i < M; i++) infile >> h_out[i]; 
	infile.close();

	return h_out;

}

template float  * loadCPUrealtxt<float> (const char *, float  * __restrict__, const int);
template double * loadCPUrealtxt<double>(const char *, double * __restrict__, const int);


