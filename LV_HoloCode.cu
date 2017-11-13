

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


//#define IDX2R(i,j,N) (((i)*(N))+(j)) //easy way to address 2D array
__global__ void fftshift_2D(cufftComplex *data, int arraysize, int row)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = threadID; i < arraysize; i += numThreads)	{
		int k=i%row;
		int j=i/row;
		
		float a = 1 - 2 * ((k + j) & 1);
		data[i].x *= a;
		data[i].y *= a;
	}
}

__device__ static __inline__ float cmagf2(float x, float y)
{
	float a, b, v, w, t;
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


////////////////////////////////
///////////////////////// GPU Kernels
//////////////////////////////

//this kernel requires fftshift
__global__ void makeKernel(float* KernelPhase, int row, int column, float* ImgProperties, float MagXscaling) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	float MagX = ImgProperties[1];
	float pixSize= ImgProperties[0];
	float nm = ImgProperties[2];
	float lambda = ImgProperties[3];


	float pixdxInv = MagX/pixSize*MagXscaling; // Magnification/pixSize
	float km = nm/lambda; // nm / lambda
														  
	for (int i = threadID; i < row*column; i += numThreads) {
		int dx = i%row;
		int dy = i/row; 

		float kdx = float( dx - row/2)*pixdxInv;
		float kdy = float( dy - row/2)*pixdxInv;
		float temp = km*km - kdx*kdx - kdy*kdy;
		KernelPhase[i]= (temp >= 0) ? sqrtf(temp) : 0;


		//This still needs quadrant swapping so this will not work in the ifft routine as is! 
		
			

	}
}


///Generates a kernel that is compatible with the non-shifted fft routine
__global__ void makeKernel_nonefftshift(float* KernelPhase, int row, int column, float* ImgProperties, float MagXscaling) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	


	float MagX = ImgProperties[1];
	float pixSize = ImgProperties[0];
	float nm = ImgProperties[2];
	float lambda = ImgProperties[3];
	float pixdxInv = MagX / pixSize*MagXscaling; // Magnification/pixSize
	float km = nm / lambda; // nm / lambda

	
	for (int i = threadID; i < row*column; i += numThreads) {
		int dx = i % row;
		int dy = i / row;
		
		dx= ((dx - row / 2)>0) ? (dx - row) : dx;
		dy= ((dy - row / 2)>0) ? (dy - row) : dy;
				
		float kdx = float(dx)*pixdxInv;
		float kdy = float(dy)*pixdxInv;
		float temp = km*km - kdx*kdx - kdy*kdy;
		KernelPhase[i] = (temp >= 0) ? sqrtf(temp) : 0;
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
				float phase = bfpPhase[k]+(kPhase[k]*zDist[j]); //multiply here already , absorb the 2*pi in there
				img3Darray[i].x = mag*cosf(phase);
				img3Darray[i].y = mag*sinf(phase);
			}
		}


__global__ void Cmplx2ReIm(cufftComplex* cmplxArray, float* reArray, float* imgArray, int size, int imgsize) {
			const int numThreads = blockDim.x * gridDim.x;
			const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
			for (int i = threadID; i < size; i += numThreads){
				int k = i/imgsize;
				reArray[i] = cmplxArray[i].x;
				imgArray[i] = cmplxArray[i].y;

			}
		}

__global__ void Cmplx2Mag(cufftComplex* cmplxArray, float* MagArray, int size, int imgsize) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = threadID; i < size; i += numThreads) {
		int k = i / imgsize;
		MagArray[i] = cmagf2(cmplxArray[i].x, cmplxArray[i].y);
		//imgArray[i] = cmplxArray[i].y;

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


			float MagXReScale = 1.0f/float(resizeRow);


		


			//////////////////////////////////////////////////
			//transfer data from host memory to GPU 
			//// idea is to avoid an expensive c++ allocation and copying values into a complex array format
			////// Almost thinking of calculating the whole Kernel in the device to avoid 2 device transfers!

			int numElements = row*column;
			size_t mem2darray = numElements*sizeof(float);

			const int BlockSizeAll = 512;
			int GridSizeKernel = (numElements + BlockSizeAll-1)/BlockSizeAll;


			float* d_kernelPhase;
			cudaMalloc((void**)&d_kernelPhase, mem2darray);

			float *d_imgProperties;
			size_t sizePrp = 4 * sizeof(float);
			cudaMalloc((void**)&d_imgProperties, sizePrp);
			cudaMemcpy(d_imgProperties, imgProperties, sizePrp, cudaMemcpyHostToDevice);

			makeKernel_nonefftshift <<<GridSizeKernel, BlockSizeAll,0,0 >>>(d_kernelPhase, row, column, d_imgProperties, MagXReScale);

			float* d_bfpMag;
			float* d_bfpPhase;
			cudaMalloc((void**)&d_bfpMag, mem2darray);
			cudaMalloc((void**)&d_bfpPhase, mem2darray);

			cudaMemcpy(d_bfpMag, h_bfpMag, mem2darray, cudaMemcpyHostToDevice);
			cudaMemcpy(d_bfpPhase, h_bfpPhase, mem2darray, cudaMemcpyHostToDevice);

			float *d_zscale;
			size_t memzsize = zrange * sizeof(float);
			cudaMalloc((void**)&d_zscale, memzsize);
			cudaMemcpy(d_zscale, zscale, memzsize, cudaMemcpyHostToDevice);

			//preallocate space for 3D array, this will be a bit costly but lets go ahead with it
			cufftComplex *d_3DiFFT;
			int size3Darray = row*column*zrange;
			size_t mem3dsize = size3Darray * sizeof(cufftComplex);
			cudaMalloc((void**)&d_3DiFFT, mem3dsize);

			//Execute Kernels
			int GridSizeTransfer = (numElements*zrange/16+BlockSizeAll-1)/BlockSizeAll;
			TransferFunction <<<GridSizeTransfer, BlockSizeAll,0,0 >>> (d_3DiFFT, d_bfpMag , d_bfpPhase, d_kernelPhase, d_zscale, size3Darray, numElements);
			
			//given that LV does not accept the cmplx number array format as any I/O I need to transform the cmplx 3D array into re and im. 
			// temporarily removed ... as the copy could be done in a single pass!
			float* d_ImgOutRe;
			float* d_ImgOutIm;
			size_t mem3dfloat = size3Darray*sizeof(float);
			cudaMalloc((void**)&d_ImgOutRe, mem3dfloat);
			cudaMalloc((void**)&d_ImgOutIm, mem3dfloat);


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
			
			//free handle , Although might be able to reuse upon the last execution
			cufftDestroy(BatchFFTPlan);


			///////////
			// FFT ends
			///////////

			//Kernel to transform into a LV happy readable array
			Cmplx2ReIm <<<GridSizeTransfer, BlockSizeAll,0,0 >>> (d_3DiFFT, d_ImgOutRe, d_ImgOutIm, size3Darray,numElements);
			
			//Copy device memory to hosts
					
			cudaMemcpy(h_ImgOutRe,d_ImgOutRe, mem3dfloat, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_ImgOutIm,d_ImgOutIm, mem3dfloat, cudaMemcpyDeviceToHost);


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




		void PropagateZ_ReturnMagnitude(float* h_bfpMag, float* h_bfpPhase,
			float* h_ImgOutMag, float* zscale, int* arraySize, float* imgProperties) {

			//Extract the size of the 2D and 3D arrays, and their respect allocation sizes
			int row = arraySize[0];
			int column = arraySize[1];
			int zrange = arraySize[2];
			int resizeRow = arraySize[3];


			float MagXReScale = 1.0f / float(resizeRow);





			//////////////////////////////////////////////////
			//transfer data from host memory to GPU 
			//// idea is to avoid an expensive c++ allocation and copying values into a complex array format
			////// Almost thinking of calculating the whole Kernel in the device to avoid 2 device transfers!

			int numElements = row*column;
			size_t mem2darray = numElements * sizeof(float);

			const int BlockSizeAll = 512;
			int GridSizeKernel = (numElements + BlockSizeAll - 1) / BlockSizeAll;


			float* d_kernelPhase;
			cudaMalloc((void**)&d_kernelPhase, mem2darray);

			float *d_imgProperties;
			size_t sizePrp = 4 * sizeof(float);
			cudaMalloc((void**)&d_imgProperties, sizePrp);
			cudaMemcpy(d_imgProperties, imgProperties, sizePrp, cudaMemcpyHostToDevice);

			makeKernel_nonefftshift << <GridSizeKernel, BlockSizeAll, 0, 0 >> >(d_kernelPhase, row, column, d_imgProperties, MagXReScale);

			float* d_bfpMag;
			float* d_bfpPhase;
			cudaMalloc((void**)&d_bfpMag, mem2darray);
			cudaMalloc((void**)&d_bfpPhase, mem2darray);

			cudaMemcpy(d_bfpMag, h_bfpMag, mem2darray, cudaMemcpyHostToDevice);
			cudaMemcpy(d_bfpPhase, h_bfpPhase, mem2darray, cudaMemcpyHostToDevice);

			float *d_zscale;
			size_t memzsize = zrange * sizeof(float);
			cudaMalloc((void**)&d_zscale, memzsize);
			cudaMemcpy(d_zscale, zscale, memzsize, cudaMemcpyHostToDevice);

			//preallocate space for 3D array, this will be a bit costly but lets go ahead with it
			cufftComplex *d_3DiFFT;
			int size3Darray = row*column*zrange;
			size_t mem3dsize = size3Darray * sizeof(cufftComplex);
			cudaMalloc((void**)&d_3DiFFT, mem3dsize);

			//Execute Kernels
			int GridSizeTransfer = (numElements*zrange / 16 + BlockSizeAll - 1) / BlockSizeAll;
			TransferFunction << <GridSizeTransfer, BlockSizeAll, 0, 0 >> > (d_3DiFFT, d_bfpMag, d_bfpPhase, d_kernelPhase, d_zscale, size3Darray, numElements);

			//given that LV does not accept the cmplx number array format as any I/O I need to transform the cmplx 3D array into re and im. 
			// temporarily removed ... as the copy could be done in a single pass!
			float* d_ImgOutMag;
			//float* d_ImgOutIm;
			size_t mem3dfloat = size3Darray * sizeof(float);
			cudaMalloc((void**)&d_ImgOutMag, mem3dfloat);
			//cudaMalloc((void**)&d_ImgOutIm, mem3dfloat);


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

			//free handle , Although might be able to reuse upon the last execution
			cufftDestroy(BatchFFTPlan);


			///////////
			// FFT ends
			///////////

			//Kernel to transform into a LV happy readable array
			Cmplx2Mag << <GridSizeTransfer, BlockSizeAll, 0, 0 >> > (d_3DiFFT, d_ImgOutMag, size3Darray, numElements);

			//Copy device memory to hosts

			cudaMemcpy(h_ImgOutMag, d_ImgOutMag, mem3dfloat, cudaMemcpyDeviceToHost);
			//cudaMemcpy(h_ImgOutIm, d_ImgOutIm, mem3dfloat, cudaMemcpyDeviceToHost);


			//deallocate CUDA memory

			cudaFree(d_bfpMag);
			cudaFree(d_bfpPhase);
			cudaFree(d_kernelPhase);
			cudaFree(d_3DiFFT);
			cudaFree(d_zscale);
			cudaFree(d_imgProperties);
			cudaFree(d_ImgOutMag);
			//cudaFree(d_ImgOutIm);

		}
			

		void TestMakeKernel3D(float* h_bfpMag, float* h_bfpPhase,
			float* h_ImgOutRe, float* h_ImgOutIm,
			float* zscale, int* arraySize, float* imgProperties) {


			//Extract the size of the 2D and 3D arrays, and their respect allocation sizes
			int row = arraySize[0];
			int column = arraySize[1];
			int zrange = arraySize[2];
			int resizeRow = arraySize[3];


			float MagXReScale = 1.0f / float(resizeRow);


			const int BlockSize = 512;
			int GridSize = 32 * 16 * 4;


			//////////////////////////////////////////////////
			//transfer data from host memory to GPU 
			//// idea is to avoid an expensive c++ allocation and copying values into a complex array format
			////// Almost thinking of calculating the whole Kernel in the device to avoid 2 device transfers!

			int numElements = row*column;
			size_t mem2darray = numElements * sizeof(float);

			float* d_kernelPhase;
			cudaMalloc((void**)&d_kernelPhase, mem2darray);

			float *d_imgProperties;
			size_t sizePrp = 4 * sizeof(float);
			cudaMalloc((void**)&d_imgProperties, sizePrp);
			cudaMemcpy(d_imgProperties, imgProperties, sizePrp, cudaMemcpyHostToDevice);

			makeKernel << <GridSize, BlockSize, 0, 0 >> >(d_kernelPhase, row, column, d_imgProperties, MagXReScale);

			float* d_bfpMag;
			float* d_bfpPhase;
			cudaMalloc((void**)&d_bfpMag, mem2darray);
			cudaMalloc((void**)&d_bfpPhase, mem2darray);

			cudaMemcpy(d_bfpMag, h_bfpMag, mem2darray, cudaMemcpyHostToDevice);
			cudaMemcpy(d_bfpPhase, h_bfpPhase, mem2darray, cudaMemcpyHostToDevice);

			float *d_zscale;
			size_t memzsize = zrange * sizeof(float);
			cudaMalloc((void**)&d_zscale, memzsize);
			cudaMemcpy(d_zscale, zscale, memzsize, cudaMemcpyHostToDevice);

			//preallocate space for 3D array, this will be a bit costly but lets go ahead with it
			cufftComplex *d_3DiFFT;
			int size3Darray = row*column*zrange;
			size_t mem3dsize = size3Darray * sizeof(cufftComplex);
			cudaMalloc((void**)&d_3DiFFT, mem3dsize);

			//given that LV does not accept the cmplx number array format as any I/O I need to transform the cmplx 3D array into re and im. 

			float* d_ImgOutRe;
			float* d_ImgOutIm;
			size_t mem3dfloat = size3Darray * sizeof(float);
			cudaMalloc((void**)&d_ImgOutRe, mem3dfloat);
			cudaMalloc((void**)&d_ImgOutIm, mem3dfloat);

			//Execute Kernels
			//TransferFunction << <GridSize, BlockSize, 0, 0 >> > (d_3DiFFT, d_bfpMag, d_bfpPhase, d_kernelPhase, d_zscale, size3Darray, numElements);

			//Kernel to transform into a LV happy readable array
			//Cmplx2ReIm << <GridSize, BlockSize, 0, 0 >> > (d_3DiFFT, d_ImgOutRe, d_ImgOutIm, size3Darray);



			//Copy device memory to host
			cudaMemcpy(h_ImgOutRe, d_ImgOutRe, mem3dfloat, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_ImgOutIm, d_ImgOutIm, mem3dfloat, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_bfpPhase, d_kernelPhase, mem2darray, cudaMemcpyDeviceToHost);

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
		