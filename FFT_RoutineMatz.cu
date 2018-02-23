#include "CudaDLL.h"
#include <stdio.h>
#include <cufft.h>
#include <cuComplex.h>
#include <device_functions.h>
#include <math.h>
#include <float.h>
///////////////////////////////
///////////// Device specific operations
//////////////////////////


__global__ void real2complex(float *dataIn, cufftComplex *dataOut, int arraysize)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = threadID; i < arraysize; i += numThreads) {
		dataOut[i].x = dataIn[i];
		dataOut[i].y = 0.0f;
	}
	
}


__global__ void C2R(cufftComplex* cmplxArray, float* reArray, float* imgArray, int size) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = threadID; i < size; i += numThreads) {
		reArray[i] = cmplxArray[i].x;
		imgArray[i] = cmplxArray[i].y;

	}
}

///////////////////////
//////////////// Executable functions 
///////////////////////


void ExtractGradients(float* h_rawImg, int* arraySize, float* imgProperties,
	float* h_ImgDxOutRe, float* h_ImgDxOutIm,
	float* h_ImgDyOutRe, float* h_ImgDyOutIm) {
	
//Declare constants
	const int row = arraySize[0];
	const int column = arraySize[1];
	const int zrange = 1; // in this case Matz is only doing one image at a time
	const int imgpropsize = arraySize[2];
	const size_t size2Darray = row*column;
	const size_t mem2Darray = size2Darray * sizeof(float);
	const size_t mem2DFFTsize = size2Darray * sizeof(cufftComplex);

 // Declare all constant regarding the Kernel execution sizes, will need to add a possibility to modify these from the LV as arguments
	const int BlockSizeAll = 512;
	const int GridSizeKernel = (size2Darray + BlockSizeAll - 1) / BlockSizeAll;

// Copy Raw Img and spatial filtering constants to GPU device
	float* d_rawImg, float* d_imgProperties;
	const size_t sizePrp = imgpropsize * sizeof(float);
	cudaMalloc((void**)&d_rawImg, mem2Darray);
	cudaMemcpy(d_rawImg, h_rawImg, mem2Darray, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_imgProperties, sizePrp);
	cudaMemcpy(d_imgProperties, imgProperties, sizePrp, cudaMemcpyHostToDevice);
	

//Img memory allocations on the GPU to hold BFP and derivates of X and Y
	cufftComplex *d_BFP;
	cufftComplex *d_GradDx;
	cufftComplex *d_GradDy;
	cudaMalloc((void**)&d_BFP, mem2DFFTsize);
	cudaMalloc((void**)&d_GradDx, mem2DFFTsize);
	cudaMalloc((void**)&d_GradDy, mem2DFFTsize);
	
	

 /////////////////////////////////////////////////////////////////////////////////////////
 ///// Prepare batch 2D FFT plan, const declaration
 /////////////////////////////////////////////////////////////////////////////////////////
	
	int istride = 1; //means every element is used in the computation
	int ostride = 1; //means every element used in the computatio is output
	int idist = row*column;
	int odist = row*column;
	int inembed[] = { row,column };
	int onembed[] = { row,column };
	const int NRANK = 2;
	int n[NRANK] = { row,column };
	int BATCH = zrange;

	cufftHandle SingleFFTPlan;
		if (cufftPlanMany(&SingleFFTPlan, NRANK, n,
		inembed, istride, idist,// *inembed, istride, idist 
		onembed, ostride, odist,// *onembed, ostride, odist 
		CUFFT_C2C, BATCH) != CUFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFT Error: Unable to create plan\n");
		return;
	}

	

	// Convert d-raw img into a complex number!
		real2complex <<<GridSizeKernel, BlockSizeAll, 0, 0 >>>(d_rawImg, d_BFP, size2Darray);

	/// Execute FFT transform in-place to go into kspace, 
		cufftExecC2C(SingleFFTPlan, d_BFP, d_BFP, CUFFT_FORWARD);

	/// Extract gradients in X and Y, frequency filtering 


	/// Inverse FFT in-place for each of the gradients
		cufftExecC2C(SingleFFTPlan, d_GradDx, d_GradDx, CUFFT_INVERSE);
		cufftExecC2C(SingleFFTPlan, d_GradDy, d_GradDy, CUFFT_INVERSE);

	//free handle , Although might be able to reuse upon the last execution
		cufftDestroy(SingleFFTPlan);
	
// Copy FFT result to output

		float *d_ImgDxOutRe; 
		float *d_ImgDxOutIm;
		float *d_ImgDyOutRe;
		float *d_ImgDyOutIm;
		cudaMalloc((void**)&d_ImgDxOutRe, mem2Darray);
		cudaMalloc((void**)&d_ImgDxOutIm, mem2Darray);
		cudaMalloc((void**)&d_ImgDyOutRe, mem2Darray);
		cudaMalloc((void**)&d_ImgDyOutIm, mem2Darray);
		

		C2R << <GridSizeTransfer, BlockSizeAll, 0, 0 >> > (d_GradDx, d_ImgDxOutRe, d_ImgDxOutIm, size2Darray);
		cudaFree(d_GradDx);
		C2R << <GridSizeTransfer, BlockSizeAll, 0, 0 >> > (d_GradDy, d_ImgDyOutRe, d_ImgDyOutIm, size2Darray);
		cudaFree(d_GradDy);

		cudaMemcpy(h_ImgDxOutRe, d_ImgDxOutRe, mem2Darray, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_ImgDxOutIm, d_ImgDxOutIm, mem2Darray, cudaMemcpyDeviceToHost);
		cudaFree(d_ImgDxOutRe);
		cudaFree(d_ImgDxOutIm);

		cudaMemcpy(h_ImgDyOutRe, d_ImgDyOutRe, mem2Darray, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_ImgDyOutIm, d_ImgDyOutIm, mem2Darray, cudaMemcpyDeviceToHost);
		cudaFree(d_ImgDyOutRe);
		cudaFree(d_ImgDyOutIm);

		//d_ImgdxOutRe
	///////////
	// FFT ends
	///////////

}

