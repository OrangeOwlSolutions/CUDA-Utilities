#include <fstream>
#include <iomanip>

#include <cuda.h>

#include "InputOutput.cuh"
#include "Utilities.cuh"

#define prec_save 10

/***********************************************/
/* SAVE INDIVIDUAL REAL CPU MATRIX TO txt FILE */
/***********************************************/
template <class T>
void saveGPUrealtxt(const T * d_in, const char *filename, const int M) {
	
	T *h_in	= (T *)malloc(M * sizeof(T));
	
	gpuErrchk(cudaMemcpy(h_in, d_in, M * sizeof(T), cudaMemcpyDeviceToHost));
	
	std::ofstream outfile;
	outfile.open(filename);
	for(int i = 0; i < M; i++) outfile << std::setprecision(prec_save) << h_in[i] << "\n"; 
	outfile.close();

}

template void saveGPUrealtxt<float> (const float  *, const char *, const int);
template void saveGPUrealtxt<double>(const double *, const char *, const int);

