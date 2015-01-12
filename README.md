# CUDA Utilities
Utilities for CUDA programming

Wrap all the CUDA API calls within ```gpuErrchk```. For example:

```
gpuErrchk(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));
```

Use ```gpuErrchk``` to check for errors in your kernels. For example:

```
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
```

If ```DEBUG``` is defined, then the CUDA error check of the kernels will be executed.

Use ```iDivUp``` to find the number of blocks to be launched by a CUDA kernel. For example:

```
BLOCKSIZE = 256;
N = 10000;       // Number of elements to be processed
kernel<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(...);
```
