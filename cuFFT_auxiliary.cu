/***************/
/* FFTSHIFT 2D */
/***************/
__global__ void fftshift_2D(float2 * __restrict__ data, const int N1, const int N2)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i < N1 && j < N2) {
		data[i*N2+j].x *= 1-2*((i+j)&1);
		data[i*N2+j].y *= 1-2*((i+j)&1);
	}
}

__global__ void fftshift_2D(double2 * __restrict__ data, const int N1, const int N2)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i < N1 && j < N2) {
		data[i*N2+j].x *= 1-2*((i+j)&1);
		data[i*N2+j].y *= 1-2*((i+j)&1);
	}
}
