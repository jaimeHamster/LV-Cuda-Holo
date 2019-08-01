//add these two to get rid of intellisense errors

#include "CudaDLL.h"
#include <stdio.h>
#include <cuComplex.h>
#include <device_functions.h>
#include <math.h>
#include <float.h>


/// Cuda function to calculate haarlike featuers

__global__ void SinglHaarCalculate(float* img3Darray, int totalsize, int imgsize, int row)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	//additional counters
	for (int i = threadID; i < totalsize; i += numThreads)
	{
		int idz = i / imgsize;
		int kimg = i % imgsize;
		int idy = kimg / row;
		int idx = kimg % row;
	
	
	
	
	}
}





//This code obtains performs Haar wavelets computations from an input 3D integral image 

// Input3D IntegralImage
// Foreground element size (x,y,z)
// Background element size (x,y,z)
// How many boxes to compute
// Scan box length


void CalculateHaarFeatures(float* h_intImage3D, float* h_HaarFeatures,
	int* h_ImgOutMag, float* zscale, int* arraySize, float* imgProperties, int* GPUspecs) {

	//Extract the size of the 2D and 3D arrays, and their respect allocation sizes
	const int row = arraySize[0];
	const int column = arraySize[1];
	const int zrange = arraySize[2];

	const int numElements = row * column;
	const int size3Darray = row * column*zrange;

	const size_t memZsize = zrange * sizeof(float);
	const size_t mem2Darray = numElements * sizeof(float);
	const size_t mem3Dsize = size3Darray * sizeof(cufftComplex);
	const size_t mem3Darray = size3Darray * sizeof(float);
	const size_t sizePrp = 5 * sizeof(float);


	//Declare all constants regarding Kernel execution sizes
	const int BlockSizeAll = GPUspecs[0];
	const int GridSizeKernel = (numElements + BlockSizeAll - 1) / BlockSizeAll;
	const int GridSizeTransfer = (size3Darray / 16 + BlockSizeAll - 1) / BlockSizeAll;

	//////////////////////////////////////////////////
	//transfer data from host memory to GPU 
	//// idea is to avoid an expensive c++ allocation and copying values into a complex array format
	////// Almost thinking of calculating the whole Kernel in the device to avoid 2 device transfers!

	float* d_kernelPhase;
	float *d_imgProperties;
	cudaMalloc((void**)&d_kernelPhase, mem2Darray);
	cudaMalloc((void**)&d_imgProperties, sizePrp);
	cudaMemcpy(d_imgProperties, imgProperties, sizePrp, cudaMemcpyHostToDevice);

	makeKernel_nonefftshift << <GridSizeKernel, BlockSizeAll, 0, 0 >> > (d_kernelPhase, row, column, d_imgProperties);

	float* d_bfpMag;
	float* d_bfpPhase;
	cudaMalloc((void**)&d_bfpMag, mem2Darray);
	cudaMalloc((void**)&d_bfpPhase, mem2Darray);
	cudaMemcpy(d_bfpMag, h_bfpMag, mem2Darray, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bfpPhase, h_bfpPhase, mem2Darray, cudaMemcpyHostToDevice);

	float *d_zscale;
	cudaMalloc((void**)&d_zscale, memZsize);
	cudaMemcpy(d_zscale, zscale, memZsize, cudaMemcpyHostToDevice);

	//preallocate space for 3D array, this will be a bit costly but lets go ahead with it
	cufftComplex *d_3DiFFT;
	cudaMalloc((void**)&d_3DiFFT, mem3Dsize);

	//Execute Kernels
	TransferFunction << <GridSizeTransfer, BlockSizeAll, 0, 0 >> > (d_3DiFFT, d_bfpMag, d_bfpPhase, d_kernelPhase, d_zscale, size3Darray, numElements);

	//deallocate CUDA memory
	cudaFree(d_bfpMag);
	cudaFree(d_bfpPhase);
	cudaFree(d_zscale);
	cudaFree(d_imgProperties);
	cudaFree(d_kernelPhase);



	//Kernel to transform into a LV happy readable array
	Cmplx2Mag << <GridSizeTransfer, BlockSizeAll, 0, 0 >> > (d_3DiFFT, d_ImgOutMag, size3Darray, numElements);

	//Copy device memory to hosts
	cudaMemcpy(h_ImgOutMag, d_ImgOutMag, mem3Darray, cudaMemcpyDeviceToHost);


	//deallocate CUDA memory



	cudaFree(d_3DiFFT);
	cudaFree(d_ImgOutMag);

}