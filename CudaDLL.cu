// includes, project

#include <cutil_inline.h>


// Labview will pass array 'h_a' ('h' stands for host), scalar 'alpha' and array size.
#define BLOCKSIZE 512 // 512 is the maximum number of threads in the block.

__global__ void ScaleMatrix_Kernel(float *d_a, float alpha, int arraySize)
{
	// Block index
	int bx = blockIdx.x;

	// Thread index
	int tx = threadIdx.x;
	int begin = blockDim.x * bx;
	int index = begin + tx;

	// copies array into shared memory, important only if threads are communicating between each other. Its not necessary here since we are only scaling vector.

	__shared__ float d_as[BLOCKSIZE];

	d_as[tx] = d_a[index];

	__syncthreads();

	// copies array back to global device memory

	d_a[index] = alpha * d_as[tx];

}



__declspec(dllexport) void ScaleMatrix(float *h_a, float alpha, int arraySize)

{
	unsigned int mem_size = sizeof(float) * arraySize;

	// allocate device memory

	float* d_a;

	cutilSafeCall(cudaMalloc((void**)&d_a, mem_size));

	// copy host memory to global device memory

	cutilSafeCall(cudaMemcpy(d_a, h_a, mem_size, cudaMemcpyHostToDevice));

	// setup execution parameters

	dim3 dimGrid(1, 1, 1);

	dim3 dimBlock(BLOCKSIZE, 1, 1); // assumes arraySize is the multiples of BLOCKSIZE! or less then a BLOCKSIZE

	// execute the kernel

	ScaleMatrix_Kernel <<< dimGrid, dimBlock >>>(d_a, alpha, arraySize);

	// copy device memory to host

	cutilSafeCall(cudaMemcpy(h_a, d_a, mem_size, cudaMemcpyDeviceToHost));

	cutilSafeCall(cudaFree(d_a));

}
