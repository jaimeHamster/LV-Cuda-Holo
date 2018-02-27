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


__global__ void FrequencyFilter(cufftComplex* BFP, cufftComplex* GradBFP, float* imgProp, int row, int column, BOOLEAN Top) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = row*column;
    const float kdr = imgProp[2];

	for (int i = threadID; i < size; i += numThreads) {
		int idx = i % row;
		int idy = i / row;

		/* represents the mask for bandpass frequency filtering*/
		int dx = (idx < (row / 2)) ? idx : (idx - row);
		int dy = (idy < (row / 2)) ? idy : (idy - row);
		float temp = kdr*kdr - dx*dx - dy*dy;

		if (Top) {
				float kdx = imgProp[0];
				float kdy = imgProp[1];

				if (idx < (row / 2)) {
					idx = idx + kdx;
				}
				else {
					idx = ((dx + kdx)>0)? dx+kdx : idx-kdx;
				}

				if (idy < (row / 2)) {
					idy = idy + kdy;
				}
				else {
					idy = ((dy + kdy)>0) ? dy + kdy : idy-kdy;
				}



			}
			else {
				float kdx = imgProp[1];
				float kdy = imgProp[0];

				if (idx < (row / 2)) {
					idx = idx + kdx;
				}
				else {
					idx = ((dx + kdx)>0) ? dx + kdx : idx - kdx;
				}

				if (idy < (row / 2)) {
					idy = idy + kdy;
				}
				else {
					idy = ((dy + kdy)>0) ? dy + kdy : idy - kdy;
				}

			}
		
		//;
		//;

		GradBFP[i].x = (temp>=0) ? BFP[idx + idy*row].x : 0;
		GradBFP[i].y = (temp>=0) ? BFP[idx + idy*row].y : 0;
		
	}
}


__global__ void ExtractGradsBFP(cufftComplex* BFP, cufftComplex* GradxBFP, cufftComplex* GradyBFP, cufftComplex* DC_BFP, float* imgProp, int row, int column) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = row*column;
	const float kdx = imgProp[0];
	const float kdy = imgProp[1];
	const float kdr = imgProp[2];
	

	for (int i = threadID; i < size; i += numThreads) {
		int idx = i % row;
		int idy = i / row;
		int iidx= idx;
		int iidy= idy;
		/* represents the mask for bandpass frequency filtering*/
		int dx = (idx < (row / 2)) ? idx : (idx - row);
		int dy = (idy < (row / 2)) ? idy : (idy - row);
		float temp = kdr*kdr - dx*dx - dy*dy;
		

		/*if (idx < (row / 2)) {
				idx = idx + kdx;
				iidx = iidx + kdy;
			}
			else {
				idx = ((dx + kdx)>0) ? dx + kdx : idx - kdx;
				iidx = ((dx + kdy)>0) ? dx + kdy : iidx - kdy;
			}

			if (idy < (row / 2)) {
				idy = idy + kdy;
				iidy = iidy + kdx;
			}
			else {
				idy = ((dy + kdy) > 0) ? dy + kdy : idy - kdy;
				iidy = ((dy + kdx) > 0) ? dy + kdx : iidy - kdx;
			}*/


			if (idx < (row / 2)) {
				idx = idx + kdx;
				iidx = iidx + kdy;
			}
			else {
				idx = ((dx + kdx)>0) ? dx + kdx : idx - kdx;
				iidx = ((dx + kdy)>0) ? dx + kdy : iidx - kdy;
			}

			if (idy < (row / 2)) {
				idy = idy + kdy;
				iidy = iidy + kdx;
			}
			else {
				idy = ((dy + kdy) > 0) ? dy + kdy : idy - kdy;
				iidy = ((dy + kdx) > 0) ? dy + kdx : iidy - kdx;
			}
		
		//;
		//;
		//Problems for kdx different than 0!
		GradyBFP[i].x = (temp >= 0) ? BFP[idx + idy*row].x : 0;
		GradyBFP[i].y = (temp >= 0) ? BFP[idx + idy*row].y : 0;
		GradxBFP[i].x = (temp >= 0) ? BFP[iidx + iidy*row].x : 0;
		GradxBFP[i].y = (temp >= 0) ? BFP[iidx + iidy*row].y : 0;
		DC_BFP[i].x = (temp >= 0) ? BFP[i].x : 0;
		DC_BFP[i].y = (temp >= 0) ? BFP[i].y : 0;

	}
}
///////////////////////
//////////////// Executable functions 
///////////////////////


void ExtractGradients(float* h_rawImg, int* arraySize, float* imgProperties,
	float* h_ImgDxOutRe, float* h_ImgDxOutIm,
	float* h_ImgDyOutRe, float* h_ImgDyOutIm, 
	float* h_ImgDCOutRe, float* h_ImgDCOutIm) {
	
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
	cufftComplex *d_DC;
	cudaMalloc((void**)&d_BFP, mem2DFFTsize);
	cudaMalloc((void**)&d_GradDx, mem2DFFTsize);
	cudaMalloc((void**)&d_GradDy, mem2DFFTsize);
	cudaMalloc((void**)&d_DC, mem2DFFTsize);
	
	

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
		//so far so good up to here

	/// Extract gradients in X and Y, frequency filtering 
		
		ExtractGradsBFP << <GridSizeKernel, BlockSizeAll, 0, 0 >> > (d_BFP, d_GradDx, d_GradDy, d_DC, d_imgProperties, row, column);
		
			
		/*BOOLEAN Top = 1;
		FrequencyFilter <<<GridSizeKernel, BlockSizeAll, 0, 0 >>> (d_BFP, d_GradDy, d_imgProperties, row, column, Top);
		Top = 0;
		FrequencyFilter <<<GridSizeKernel, BlockSizeAll, 0, 0 >>> (d_BFP, d_GradDx, d_imgProperties, row, column, Top);*/



	/// Inverse FFT in-place for each of the gradients 
		// can think of batching this!
		cufftExecC2C(SingleFFTPlan, d_GradDx, d_GradDx, CUFFT_INVERSE);
		cufftExecC2C(SingleFFTPlan, d_GradDy, d_GradDy, CUFFT_INVERSE);
		cufftExecC2C(SingleFFTPlan, d_DC, d_DC, CUFFT_INVERSE);

	//free handle , Although might be able to reuse upon the last execution
		cufftDestroy(SingleFFTPlan);
	
// Copy FFT result to output

		float *d_ImgDxOutRe; 
		float *d_ImgDxOutIm;
		float *d_ImgDyOutRe;
		float *d_ImgDyOutIm;
		float *d_ImgDCOutRe;
		float *d_ImgDCOutIm;
		cudaMalloc((void**)&d_ImgDxOutRe, mem2Darray);
		cudaMalloc((void**)&d_ImgDxOutIm, mem2Darray);
		cudaMalloc((void**)&d_ImgDyOutRe, mem2Darray);
		cudaMalloc((void**)&d_ImgDyOutIm, mem2Darray);
		cudaMalloc((void**)&d_ImgDCOutRe, mem2Darray);
		cudaMalloc((void**)&d_ImgDCOutIm, mem2Darray);
		

		C2R << <GridSizeKernel, BlockSizeAll, 0, 0 >> > (d_GradDx, d_ImgDxOutRe, d_ImgDxOutIm, size2Darray);
		C2R << <GridSizeKernel, BlockSizeAll, 0, 0 >> > (d_GradDy, d_ImgDyOutRe, d_ImgDyOutIm, size2Darray);
		C2R << <GridSizeKernel, BlockSizeAll, 0, 0 >> > (d_DC, d_ImgDCOutRe, d_ImgDCOutIm, size2Darray);
		cudaFree(d_GradDx);
		cudaFree(d_GradDy);
		cudaFree(d_DC);
		cudaFree(d_BFP);


		cudaMemcpy(h_ImgDxOutRe, d_ImgDxOutRe, mem2Darray, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_ImgDxOutIm, d_ImgDxOutIm, mem2Darray, cudaMemcpyDeviceToHost);
		cudaFree(d_ImgDxOutRe);
		cudaFree(d_ImgDxOutIm);

		cudaMemcpy(h_ImgDyOutRe, d_ImgDyOutRe, mem2Darray, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_ImgDyOutIm, d_ImgDyOutIm, mem2Darray, cudaMemcpyDeviceToHost);
		cudaFree(d_ImgDyOutRe);
		cudaFree(d_ImgDyOutIm);

		cudaMemcpy(h_ImgDCOutRe, d_ImgDCOutRe, mem2Darray, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_ImgDCOutIm, d_ImgDCOutIm, mem2Darray, cudaMemcpyDeviceToHost);
		cudaFree(d_ImgDCOutRe);
		cudaFree(d_ImgDCOutIm);


		//exporting is correct
		//d_ImgdxOutRe
	///////////
	// FFT ends
	///////////

}

