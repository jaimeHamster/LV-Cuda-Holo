

#include "CudaDLL.h"
#include <stdio.h>
#include <cufft.h>
#include <cuComplex.h>

///////////////////////////////
///////////// Device specific operations
//////////////////////////

__device__ static __inline__ float cmagf(float x, float y)
{
	float a,b,v, w, t;
	a = fabsf(x);
	b = fabsf(y);
	if (a > b) {
		v = a;
		w = b;
	}
	else {
		v = b;
		w = a;
	}
	t = w / v;
	t = 1.0f + t * t;
	t = v * sqrtf(t);
	if ((v == 0.0f) || (v > 3.402823466e38f) || (w > 3.402823466e38f)) {
		t = v + w;
	}
	return t;
}

__host__ __device__ float cuPhase(cufftComplex a) {
	float phaseOut;
	phaseOut = atan2f(a.y, a.x);
	return phaseOut;
}



////////////////////////////////
///////////////////////// GPU Kernels
//////////////////////////////
__global__ void makeKernel(float* KernelPhase, int row, int column, float* ImgProperties, float MagXscaling) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	const float pixdxInv = ImgProperties[1] / ImgProperties[0]*MagXscaling; // Magnification/pixSize
	const float km = ImgProperties[2] / ImgProperties[3]; // nm / lambda
														  
	for (int i = threadID; i < row*column; i += numThreads) {
		int dx = i%row;
		int dy = i%column;
		float kdx = (dx / row - 0.5f) *pixdxInv;
		float kdy = (dy / column - 0.5f)*pixdxInv;
		float temp = km*km - kdx*kdx - kdy*kdy;
		KernelPhase[i] = (temp >= 0) ? sqrtf(temp) : 0;
	}
}


		__global__ void ConvertCmplx2Polar(float* inRe, float* inIm, float* mag, float* phase, int size) {
			const int numThreads = blockDim.x * gridDim.x;
			const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
			for (int i = threadID; i < size; i += numThreads)
			{
				phase[i] = atan2f(inIm[i], inRe[i]);
				mag[i] = cmagf(inIm[i], inRe[i]);
			}
		}


		__global__ void TransferFunction(cufftComplex* img3Darray, float* bfpMag, float* bfpPhase, float* kPhase, float* zDist, int totalsize, int imgsize)
		{
			const int numThreads = blockDim.x * gridDim.x;
			const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

			//additional counters
			for (int i = threadID; i < totalsize; i += numThreads)
			{
				int j = i / imgsize; 
				int k = i % imgsize;
				float mag = bfpMag[k];
				float phase = bfpMag[k]+(kPhase[k]*zDist[j]); //multiply here already , absorb the 2*pi in there
				
				//add the result of above to the 3D array
				img3Darray[i].x = mag*cosf(phase);
				img3Darray[i].y = mag*sinf(phase);
			}
		}


		__global__ void Cmplx2ReIm(cufftComplex* cmplxArray, float* reArray, float* imgArray, int size) {
			const int numThreads = blockDim.x * gridDim.x;
			const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
			for (int i = threadID; i < size; i += numThreads){
				reArray[i] = cmplxArray[i].x;
				imgArray[i] = cmplxArray[i].y;

			}
		}



		////////////////////////////////////////////////
		//////////////// FUnction to compile into DLL
		////////////////////////////////////////////////



		void PropagateZslices(float* h_bfpMag, float* h_bfpPhase,
			float* h_ImgOutRe, float* h_ImgOutIm,
			float* zscale, int* arraySize, float* imgProperties){

			//Extract the size of the 2D and 3D arrays, and their respect allocation sizes
			int row = arraySize[0];
			int column = arraySize[1];
			int zrange = arraySize[2];
			int resizeRow = arraySize[3];


			float MagXReScale = resizeRow / row;


			const int BlockSize = 512;
			int GridSize = 32 * 16 * 4;


			//////////////////////////////////////////////////
			//transfer data from host memory to GPU 
			//// idea is to avoid an expensive c++ allocation and copying values into a complex array format
			////// Almost thinking of calculating the whole Kernel in the device to avoid 2 device transfers!




			int numElements = row*column;
			size_t mem2darray = numElements*sizeof(float);

			float* d_kernelPhase;
			float* d_bfpMag;
			float* d_bfpPhase;

			cudaMalloc((void**)&d_kernelPhase, mem2darray);
			cudaMalloc((void**)&d_bfpMag, mem2darray);
			cudaMalloc((void**)&d_bfpPhase, mem2darray);

			cudaMemcpy(d_bfpMag, h_bfpMag, mem2darray, cudaMemcpyHostToDevice);
			cudaMemcpy(d_bfpPhase, h_bfpPhase, mem2darray, cudaMemcpyHostToDevice);

			float *d_zscale;
			size_t memzsize = zrange * sizeof(float);
			cudaMalloc((void**)&d_zscale, memzsize);
			cudaMemcpy(d_zscale, zscale, memzsize, cudaMemcpyHostToDevice);

			float *d_imgProperties;
			size_t sizePrp = 4 * sizeof(float);
			cudaMalloc((void**)&d_imgProperties, sizePrp);
			cudaMemcpy(d_imgProperties, imgProperties, sizePrp, cudaMemcpyHostToDevice);



			//preallocate space for 3D array, this will be a bit costly but lets go ahead with it
			cufftComplex *d_3DiFFt;
			int size3Darray = row*column*zrange;
			size_t mem3dsize = size3Darray * sizeof(cufftComplex);
			cudaMalloc((void**)&d_3DiFFt, mem3dsize);

			//given that LV does not accept the cmplx number array format as any I/O I need to transform the cmplx 3D array into re and im. 

			float* d_ImgOutRe;
			float* d_ImgOutIm;
			size_t mem3dfloat = size3Darray*sizeof(float);
			cudaMalloc((void**)&d_ImgOutRe, mem3dfloat);
			cudaMalloc((void**)&d_ImgOutIm, mem3dfloat);

			//Execute Kernels
			makeKernel << <GridSize, BlockSize >> >(d_kernelPhase, row, column, d_imgProperties, MagXReScale);
			TransferFunction << <GridSize, BlockSize >> > (d_3DiFFT, d_bfpMag , d_bfpPhase, d_kernelPhase, d_zscale, size3Darray, numElements);

			/////////////////////////////////////////////////////////////////////////////////////////
			///// Prepare batch 2D FFT plan, const declaration
			/////////////////////////////////////////////////////////////////////////////////////////
			/* Create a batched 2D plan, or batch FFT , need to declare when each image begins! */
			int istride = 1; //means every element is used in the computation
			int ostride = 1; //means every element used in the computatio is output
			int idist = row*column;
			int odist = row*column;
			int inembed[] = { row,column };
			int onembed[] = { row,column };
			const int NRANK = 2;
			int n[NRANK] = { row,column };
			int BATCH = zrange;

			cufftHandle BatchFFTPlan;

			if (cufftPlanMany(&BatchFFTPlan, NRANK, n,
				inembed, istride, idist,// *inembed, istride, idist 
				onembed, ostride, odist,// *onembed, ostride, odist 
				CUFFT_C2C, BATCH) != CUFFT_SUCCESS)
			{
				fprintf(stderr, "CUFFT Error: Unable to create plan\n");
				return;
			}


			//////// Execute the transform in-place
			if (cufftExecC2C(BatchFFTPlan, d_3DiFFT, d_3DiFFT, CUFFT_INVERSE) != CUFFT_SUCCESS) {
				fprintf(stderr, "CUFFT Error: Failed to execute plan\n");
				return;
			}


			/* ////////// Execute the transform out-of-place */
			/*cufftComplex *d_3Dimg;
			cudaMalloc((void**)&d_3Dimg, mem3dsize);

			if (cufftExecC2C(BatchFFTPlan, d_3DKernel, d_3Dimg, CUFFT_INVERSE) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT Error: Failed to execute plan\n");
			return;
			}
			*/

			//free handle , Although might be able to reuse upon the last execution
			cufftDestroy(BatchFFTPlan);


			///////////
			// FFT ends
			///////////

			//Kernel to transform into a LV happy readable array
			Cmplx2ReIm << <GridSize, BlockSize >> > (d_3DiFFT, d_ImgOutRe, d_ImgOutIm, size3Darray);



			//Copy device memory to host
			cudaMemcpy(h_ImgOutRe, d_ImgOutRe, mem3dfloat, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_ImgOutRe, d_ImgOutIm, mem3dfloat, cudaMemcpyDeviceToHost);


			//deallocate CUDA memory
			
			cudaFree(d_bfpMag);
			cudaFree(d_bfpPhase);
			cudaFree(d_kernelPhase);
			cudaFree(d_3DiFFT);
			cudaFree(d_zscale);
			cudaFree(d_imgProperties);
			cudaFree(d_ImgOutRe);
			cudaFree(d_ImgOutIm);

		}
			