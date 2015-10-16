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

/**************************************************/
/* SAVE INDIVIDUAL COMPLEX GPU MATRIX TO txt FILE */
/**************************************************/
template <class T>
void saveGPUcomplextxt(const T * d_in, const char *filename, const int M) {
	
	T *h_in	= (T *)malloc(M * sizeof(T));
	
	gpuErrchk(cudaMemcpy(h_in, d_in, M * sizeof(T), cudaMemcpyDeviceToHost));
	
	std::ofstream outfile;
	outfile.open(filename);
	for(int i = 0; i < M; i++) { 
		//printf("%f %f\n", h_in[i].c.x, h_in[i].c.y);
		outfile << std::setprecision(prec_save) << h_in[i].c.x << "\n"; outfile << std::setprecision(prec_save) << h_in[i].c.y << "\n";
	}
	outfile.close();

}

void saveGPUcomplextxt(const float2 * d_in, const char *filename, const int M) {
	
	float2 *h_in	= (float2 *)malloc(M * sizeof(float2));
	
	gpuErrchk(cudaMemcpy(h_in, d_in, M * sizeof(float2), cudaMemcpyDeviceToHost));
	
	std::ofstream outfile;
	outfile.open(filename);
	for(int i = 0; i < M; i++) { 
		//printf("%f %f\n", h_in[i].c.x, h_in[i].c.y);
		outfile << std::setprecision(prec_save) << h_in[i].x << "\n"; outfile << std::setprecision(prec_save) << h_in[i].y << "\n";
	}
	outfile.close();

}

void saveGPUcomplextxt(const double2 * d_in, const char *filename, const int M) {
	
	double2 *h_in	= (double2 *)malloc(M * sizeof(double2));
	
	gpuErrchk(cudaMemcpy(h_in, d_in, M * sizeof(double2), cudaMemcpyDeviceToHost));
	
	std::ofstream outfile;
	outfile.open(filename);
	for(int i = 0; i < M; i++) { 
		//printf("%f %f\n", h_in[i].c.x, h_in[i].c.y);
		outfile << std::setprecision(prec_save) << h_in[i].x << "\n"; outfile << std::setprecision(prec_save) << h_in[i].y << "\n";
	}
	outfile.close();

}

template void saveGPUcomplextxt<float2_> (const float2_  *, const char *, const int);
template void saveGPUcomplextxt<double2_>(const double2_ *, const char *, const int);

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

/****************************************************/
/* LOAD INDIVIDUAL REAL MATRIX FROM txt FILE TO GPU */
/****************************************************/
// --- Load individual real matrix from txt file
template <class T>
T * loadGPUrealtxt(const char *filename, T * __restrict__ d_out, const int M) {
		
	T *h_out	= (T *)malloc(M * sizeof(T));
	//T *d_out;	gpuErrchk(cudaMalloc(&d_out, M * sizeof(T)));
	gpuErrchk(cudaMalloc((void**)&d_out, M * sizeof(T)));

	std::ifstream infile;
	infile.open(filename);
	for(int i = 0; i < M; i++) infile >> h_out[i]; 
	infile.close();

	gpuErrchk(cudaMemcpy(d_out, h_out, M * sizeof(T), cudaMemcpyHostToDevice));
	
	return d_out;

}

template float  * loadGPUrealtxt<float> (const char *, float  * __restrict__, const int);
template double * loadGPUrealtxt<double>(const char *, double * __restrict__, const int);

