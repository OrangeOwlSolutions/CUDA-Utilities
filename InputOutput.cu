#include <fstream>
#include <iomanip>

#include <cuda.h>

#include "InputOutput.cuh"
#include "BBComplex.h"
#include "Utilities.cuh"

#define prec_save 10

/***********************************************/
/* SAVE INDIVIDUAL REAL GPU MATRIX TO txt FILE */
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

/**************************************************/
/* SAVE INDIVIDUAL COMPLEX GPU MATRIX TO txt FILE */
/**************************************************/
template <class T>
void saveGPUcomplextxt(const T * d_in, const char *filename, const int M) {
	
	T *h_in	= (T *)malloc(M * sizeof(T));
	
	gpuErrchk(cudaMemcpy(h_in, d_in, M * sizeof(T), cudaMemcpyDeviceToHost));
	
	std::ofstream outfile;
	outfile.open(filename);
	for(int i = 0; i < M; i++) { outfile << std::setprecision(prec_save) << h_in[i].c.x << "\n"; outfile << std::setprecision(prec_save) << h_in[i].c.y << "\n";}
	outfile.close();

}

template void saveGPUcomplextxt<float2_> (const float2_  *, const char *, const int);
template void saveGPUcomplextxt<double2_>(const double2_ *, const char *, const int);

/****************************************************/
/* LOAD INDIVIDUAL REAL MATRIX FROM txt FILE TO GPU */
/****************************************************/
// --- Load individual real matrix from txt file
template <class T>
T * loadGPUrealtxt(const char *filename, T * __restrict__ d_out, const int M) {
		
	T *h_out	= (T *)malloc(M * sizeof(T));
	//T *d_out;	gpuErrchk(cudaMalloc(&d_out, M * sizeof(T)));
	gpuErrchk(cudaMalloc(&d_out, M * sizeof(T)));

	std::ifstream infile;
	infile.open(filename);
	for(int i = 0; i < M; i++) infile >> h_out[i]; 
	infile.close();

	gpuErrchk(cudaMemcpy(d_out, h_out, M * sizeof(T), cudaMemcpyHostToDevice));
	
	return d_out;

}

template float  * loadGPUrealtxt<float> (const char *, float  * __restrict__, const int);
template double * loadGPUrealtxt<double>(const char *, double * __restrict__, const int);

