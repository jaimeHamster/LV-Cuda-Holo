// customDllFunctions.cu

//////////////////////////
// Template to write .dlls 
//////////////////////////

/* Include the following directories for the program to run appropriately:
///////////////////////
in the VC++ directories:

	$(VC_IncludePath);
	$(WindowsSDK_IncludePath);
	C:\ProgramData\NVIDIA Corporation\CUDA Samples\v9.0\common\inc;
	$(CUDA_INC_PATH)
	C:\Program Files\National Instruments\LabVIEW 2015\cintools

////////////////////////
CUDA/C/C++ directories:

	./
	../../common/inc
	$(CudaToolkitDir)/include

////////////////////////////////
Linker/General include libraries:
	cudart.lib

//changed the target machine platform from 32 to 64 bit
*/


#include "CudaDLL.h"
#include <stdio.h>
#include <cufft.h>
#include <cuComplex.h>



////////////////////////////////////////////////////////////////////////////////
// Complex operations, 
////////////////////////////////////////////////////////////////////////////////


__host__ __device__ float cuPhase(cufftComplex a) {
	float phaseOut;
	phaseOut = atan2f(a.y, a.x);
	return phaseOut;
}



///////////////////////////////////////////////////////////////
////// CUDA Kernels go Here, which are then called in the GPU DLL FUnctions
///////////////////////////////////////////////////////////////
//Simple array scaling to test for GPU proper operation.
__global__ void ScaleArray(float *d_a, float alpha, int arraySize)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	float temp;

	for (int i = threadID; i < arraySize; i += numThreads)
	{
		temp = d_a[i];
		d_a[i] = alpha*temp;
	}
}

// Complex 2scale
__global__ void ComplexScale2(cufftComplex* a, int size, float scale)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	
	//its imperative that operations are performed on the individual elements,can not appyling the scaling simultaneously .x/.y
	for (int i = threadID; i < size; i += numThreads)
	{
		a[i].x = scale*a[i].x;
		a[i].y = scale*a[i].y;
	}
	
}

// Create a 3D complex array from multiplication of a 2D array by 1D array
//could be some form of slow down in the code if it is done with integer division / modulo operations
//although it seems it is the only easy way to keep the threads parallelizable

__global__ void Create3DKernel(cufftComplex* KernelOut, cufftComplex* KernelIn, float* zrange, int size, int imgsize)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	//its imperative that operations are performed on the individual elements,can not appyling the scaling simultaneously .x/.y
	for (int i = threadID; i < size; i += numThreads)
	{
		int j = i / imgsize;
		int k = i % imgsize;
		KernelOut[i].x = zrange[j]*KernelIn[k].x;
		KernelOut[i].y = zrange[j]*KernelIn[k].y;
	}

}

//Need to multiply in phase land and then convert back into .re and .im
__global__ void Create3DKernelPhaseMultiply(cufftComplex* KernelOut, cufftComplex* KernelIn, float* zscale, int size, int imgsize)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadID; i < size; i += numThreads)
	{
		int j = i / imgsize;
		int k = i % imgsize;
		float mag = cuCabsf(KernelIn[k]);
		float phase = cuPhase(KernelIn[k])*zscale[j]; //multiply here already , absorb the 2*pi in there
		KernelOut[i].x = mag*cosf(phase);
		KernelOut[i].y = mag*sinf(phase);
		//add matrix multiplication here if desired!
	
	}
}

/////////////////////////////////////////////////////////////////
//////// Kernels to be most likely used for digital holography
////////////////////////////////////////////////////////////////////////

//Need to multiply in phase land and then convert back into .re and .im
__global__ void Create3DTransferFunction(cufftComplex* KernelOut, cufftComplex* KernelIn, cufftComplex* bfpIn, float* zscale, int size, int imgsize)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	//additional counters
	for (int i = threadID ; i < size; i += numThreads)
	{
		int j = i / imgsize;
		int k = i % imgsize;
		float mag = cuCabsf(KernelIn[k]);
		float phase = cuPhase(KernelIn[k])*zscale[j]; //multiply here already , absorb the 2*pi in there

		cufftComplex tempHolder;
		//converting from polar to re/im coordinate system
		tempHolder.x = mag*cosf(phase);
		tempHolder.y = mag*sinf(phase);

		//performing matrix multiplication on an index basis between bfpIn and Kernel
		cufftComplex tempMult;
		tempMult = cuCmulf(tempHolder, bfpIn[k]);

		//add the result of above to the 3D array
		KernelOut[i].x = tempMult.x;
		KernelOut[i].y = tempMult.y;
			}
}



//////////////////////////////////////////////////////////////////
////// Functions to be called via the DLL  go HERE
/////////////////////////////////////////////////////////////////


void Make3DTransform_CSG(float* h_KernelRE, float* h_KernelIm,
	float* h_bfpRe, float* h_bfpIm,
	float* h_KernelModRe, float* h_KernelModIm,
	float* zscale, int* arraySize) {

	//Extract the size of the 2D and 3D arrays, and their respect allocation sizes
	int row = arraySize[0];
	int column = arraySize[1];
	int zrange = arraySize[2];
	int numElements = row*column;
	int size3Darray = row*column*zrange;


	//transfer data from h_KernelRe and Imaginary to C++ pointer
	// As far as I know the only intelligent way to do this without going insane is to toss it into a FOR loop	
	cufftComplex *h_Kernel;
	h_Kernel = new cufftComplex[numElements]; // reserves memory for kernel

	for (int i = 0; i < numElements; i++) {
		h_Kernel[i].x = h_KernelRE[i];
		h_Kernel[i].y = h_KernelIm[i];
	}

	//transfer BFP into complex c++ notation
	cufftComplex *h_bfpIn;
	h_bfpIn = new cufftComplex[numElements]; // reserves memory for kernel

	for (int i = 0; i < numElements; i++) {
		h_bfpIn[i].x = h_bfpRe[i];
		h_bfpIn[i].y = h_bfpIm[i];
	}


	//transfer data from host memory to GPU 
	cufftComplex *d_Kernel;
	size_t memsize = numElements * sizeof(cufftComplex);
	cudaMalloc((void**)&d_Kernel, memsize);
	cudaMemcpy(d_Kernel, h_Kernel, memsize, cudaMemcpyHostToDevice);

	cufftComplex *d_bfpIn;
	cudaMalloc((void**)&d_bfpIn, memsize);
	cudaMemcpy(d_bfpIn, h_bfpIn, memsize, cudaMemcpyHostToDevice);

	float *d_zscale;
	size_t memzsize = zrange * sizeof(float);
	cudaMalloc((void**)&d_zscale, memzsize);
	cudaMemcpy(d_zscale, zscale, memzsize, cudaMemcpyHostToDevice);

	//preallocate space for 3D array, this will be a bit costly but lets go ahead with it
	cufftComplex *d_3DKernel;
	size_t mem3dsize = size3Darray * sizeof(cufftComplex);
	cudaMalloc((void**)&d_3DKernel, mem3dsize);
	//cudaMemset(d_3DKernel, 0, mem3dsize);

	//Execute Kernel
	Create3DTransferFunction <<<32, 256 >>> (d_3DKernel, d_Kernel, d_bfpIn, d_zscale, size3Darray, numElements);

	//Copy device memory to host
	cufftComplex *h_3DKernel;
	h_3DKernel = new cufftComplex[size3Darray];
	cudaMemcpy(h_3DKernel, d_3DKernel, mem3dsize, cudaMemcpyDeviceToHost);

	//deallocate CUDA memory
	cudaFree(d_bfpIn);
	cudaFree(d_Kernel);
	cudaFree(d_3DKernel);
	cudaFree(d_zscale);

	//transfer the complex2d array back as a processed individual re and imaginary 2d arrays
	// i think it is prudent that I allocate the 2D space in the dll program
	for (int i = 0; i < size3Darray; i++) {
		h_KernelModRe[i] = h_3DKernel[i].x;
		h_KernelModIm[i] = h_3DKernel[i].y;
	}

	//deallocate Host memory
	delete[] h_Kernel;
	delete[] h_3DKernel;

}


void Propagate3Dz_CSG(float* h_KernelRE, float* h_KernelIm,
	float* h_bfpRe, float* h_bfpIm,
	float* h_ImgOutRe, float* h_ImgOutIm,
	float* zscale, int* arraySize) {

	//Extract the size of the 2D and 3D arrays, and their respect allocation sizes
	int row = arraySize[0];
	int column = arraySize[1];
	int zrange = arraySize[2];
	int numElements = row*column;
	int size3Darray = row*column*zrange;


	//transfer data from h_KernelRe and Imaginary to C++ pointer
	// As far as I know the only intelligent way to do this without going insane is to toss it into a FOR loop	
	cufftComplex *h_Kernel;
	h_Kernel = new cufftComplex[numElements]; // reserves memory for kernel

	for (int i = 0; i < numElements; i++) {
		h_Kernel[i].x = h_KernelRE[i];
		h_Kernel[i].y = h_KernelIm[i];
	}

	//transfer BFP into complex c++ notation
	cufftComplex *h_bfpIn;
	h_bfpIn = new cufftComplex[numElements]; // reserves memory for kernel

	for (int i = 0; i < numElements; i++) {
		h_bfpIn[i].x = h_bfpRe[i];
		h_bfpIn[i].y = h_bfpIm[i];
	}


	//transfer data from host memory to GPU 
	cufftComplex *d_Kernel;
	size_t memsize = numElements * sizeof(cufftComplex);
	cudaMalloc((void**)&d_Kernel, memsize);
	cudaMemcpy(d_Kernel, h_Kernel, memsize, cudaMemcpyHostToDevice);

	cufftComplex *d_bfpIn;
	cudaMalloc((void**)&d_bfpIn, memsize);
	cudaMemcpy(d_bfpIn, h_bfpIn, memsize, cudaMemcpyHostToDevice);

	float *d_zscale;
	size_t memzsize = zrange * sizeof(float);
	cudaMalloc((void**)&d_zscale, memzsize);
	cudaMemcpy(d_zscale, zscale, memzsize, cudaMemcpyHostToDevice);

	//preallocate space for 3D array, this will be a bit costly but lets go ahead with it
	cufftComplex *d_3DKernel;
	size_t mem3dsize = size3Darray * sizeof(cufftComplex);
	cudaMalloc((void**)&d_3DKernel, mem3dsize);
	//cudaMemset(d_3DKernel, 0, mem3dsize);

	//Execute Kernel
	Create3DTransferFunction << <32, 256 >> > (d_3DKernel, d_Kernel, d_bfpIn, d_zscale, size3Darray, numElements);

	////////
	// FFT goes here
	////////




	////////
	// FFT ends
	///////



	//Copy device memory to host
	cufftComplex *h_3DKernel;
	h_3DKernel = new cufftComplex[size3Darray];
	cudaMemcpy(h_3DKernel, d_3DKernel, mem3dsize, cudaMemcpyDeviceToHost);

	//deallocate CUDA memory
	cudaFree(d_bfpIn);
	cudaFree(d_Kernel);
	cudaFree(d_3DKernel);
	cudaFree(d_zscale);

	//transfer the complex2d array back as a processed individual re and imaginary 2d arrays
	// i think it is prudent that I allocate the 2D space in the dll program
	for (int i = 0; i < size3Darray; i++) {
		h_ImgOutRe[i] = h_3DKernel[i].x;
		h_ImgOutIm[i] = h_3DKernel[i].y;
	}

	//deallocate Host memory
	delete[] h_Kernel;
	delete[] h_3DKernel;

}










/////////////////////////////////////////////////////////
/////
//// Basic functions to test general functionality of the program/ GPU
/////
///////////////////////////////////////

/* Add two integers */
void myGPUfunction(const char *s) {
	printf("Hello %s\n", s);
}

//Approach on how to deal with 1D arrays
void scaleArrayGPU(float *h_a, float alpha, int arraySize)
{
	//Allocate GPU memory
	float *d_a;
	size_t memsize = arraySize * sizeof(float);

	//Allocate device memory
	cudaMalloc((void**)&d_a, memsize);

	//Copy data from host to device
	cudaMemcpy(d_a, h_a, memsize, cudaMemcpyHostToDevice);

	//Execute Kernel
	ScaleArray << <1, 32 >> > (d_a, alpha, arraySize);

	//Copy device memory to host
	cudaMemcpy(h_a, d_a, memsize, cudaMemcpyDeviceToHost);

	//deallocate CUDA memory
	cudaFree(d_a);
}

//Approach to deal with 2D arrays
void scaleArray2DGPU(float* h_2a, float alpha, int* array2Dsize) {
	//Allocate GPU memory
	float *d_2a;
	int row = array2Dsize[0];
	int column = array2Dsize[1];
	size_t memsize = row*column* sizeof(float);

	//Allocate device memory
	cudaMalloc((void**)&d_2a, memsize);

	//Copy data from host to device
	cudaMemcpy(d_2a, h_2a, memsize, cudaMemcpyHostToDevice);

	//Execute Kernel
	int sizeArray = row*column;
	ScaleArray << <1, 32 >> > (d_2a, alpha, sizeArray);

	//Copy device memory to host
	cudaMemcpy(h_2a, d_2a, memsize, cudaMemcpyDeviceToHost);

	//deallocate CUDA memory
	cudaFree(d_2a);
}

//import two different float arrays to get around the stupid complex nomenclature and pointer/handle shit

void scaleArrayGPU_CSG(float* h_KernelRE, float* h_KernelIm, float* h_KernelModRe, float* h_KernelModIm, float ascale, int* arraySize) {

	//Extract the size of the 2D and 3D arrays, and their respect allocation sizes
	int row = arraySize[0];
	int column = arraySize[1];
	int zrange = arraySize[2];
	int numElements = row*column;
	//int size3Darray = row*column*zrange;


	//transfer data from h_KernelRe and Imaginary to C++ pointer
	// As far as I know the only intelligent way to do this without going insane is to toss it into a FOR loop	
	cufftComplex *h_Kernel;
	h_Kernel = new cufftComplex[numElements]; // reserves memory for kernel
											  //*h_Kernel = (cufftComplex*) malloc(memsize);
											  //This is a pure C approach

	for (int i = 0; i < numElements; i++) {
		h_Kernel[i].x = h_KernelRE[i];
		h_Kernel[i].y = h_KernelIm[i];
	}

	//transfer data from host memory to GPU 
	cufftComplex *d_Kernel;
	size_t memsize = numElements * sizeof(cufftComplex);
	cudaMalloc((void**)&d_Kernel, memsize);
	cudaMemcpy(d_Kernel, h_Kernel, memsize, cudaMemcpyHostToDevice);

	//Execute Kernel
	ComplexScale2 << <32, 32 >> > (d_Kernel, numElements, ascale);

	//Copy device memory to host
	cudaMemcpy(h_Kernel, d_Kernel, memsize, cudaMemcpyDeviceToHost);

	//deallocate CUDA memory
	cudaFree(d_Kernel);

	//transfer the complex2d array back as a processed individual re and imaginary 2d arrays
	// i think it is prudent that I allocate the 2D space in the dll program
	for (int i = 0; i < numElements; i++) {
		h_KernelModRe[i] = h_Kernel[i].x;
		h_KernelModIm[i] = h_Kernel[i].y;
	}

	//deallocated Host memory
	delete[] h_Kernel;
}


void generate3DKernel_CSG(float* h_KernelRE, float* h_KernelIm, float* h_KernelModRe, float* h_KernelModIm, float* zscale, int* arraySize) {

	//Extract the size of the 2D and 3D arrays, and their respect allocation sizes
	int row = arraySize[0];
	int column = arraySize[1];
	int zrange = arraySize[2];
	int numElements = row*column;
	int size3Darray = row*column*zrange;


	//transfer data from h_KernelRe and Imaginary to C++ pointer
	// As far as I know the only intelligent way to do this without going insane is to toss it into a FOR loop	
	cufftComplex *h_Kernel;
	h_Kernel = new cufftComplex[numElements]; // reserves memory for kernel

	for (int i = 0; i < numElements; i++) {
		h_Kernel[i].x = h_KernelRE[i];
		h_Kernel[i].y = h_KernelIm[i];
	}

	//transfer data from host memory to GPU 
	cufftComplex *d_Kernel;
	size_t memsize = numElements * sizeof(cufftComplex);
	cudaMalloc((void**)&d_Kernel, memsize);
	cudaMemcpy(d_Kernel, h_Kernel, memsize, cudaMemcpyHostToDevice);

	float *d_zscale;
	size_t memzsize = zrange * sizeof(float);
	cudaMalloc((void**)&d_zscale, memzsize);
	cudaMemcpy(d_zscale, zscale, memzsize, cudaMemcpyHostToDevice);

	//preallocate space for 3D array, this will be a bit costly but lets go ahead with it
	cufftComplex *d_3DKernel;
	size_t mem3dsize = size3Darray * sizeof(cufftComplex);
	cudaMalloc((void**)&d_3DKernel, mem3dsize);
	//cudaMemset(d_3DKernel, 0, mem3dsize);

	//Execute Kernel
	Create3DKernel << <32, 256 >> > (d_3DKernel, d_Kernel, d_zscale, size3Darray, numElements);

	//Copy device memory to host
	cufftComplex *h_3DKernel;
	h_3DKernel = new cufftComplex[size3Darray];
	cudaMemcpy(h_3DKernel, d_3DKernel, mem3dsize, cudaMemcpyDeviceToHost);

	//deallocate CUDA memory
	cudaFree(d_Kernel);
	cudaFree(d_3DKernel);
	cudaFree(d_zscale);

	//transfer the complex2d array back as a processed individual re and imaginary 2d arrays
	// i think it is prudent that I allocate the 2D space in the dll program
	for (int i = 0; i < size3Darray; i++) {
		h_KernelModRe[i] = h_3DKernel[i].x;
		h_KernelModIm[i] = h_3DKernel[i].y;
	}

	//deallocate Host memory
	delete[] h_Kernel;
	delete[] h_3DKernel;

}

/*
//////////////////////////////////
Common errors:
1)If no link , the vi has to be closed otherwise can not be modified upon compile
Not a valid win32 application: Ignore this error !! it is a .dll not an .exe

2) if all of the sudden the LV vi are not responding and output shit, the likelihood is high that the GPU is hung, so I had to restart the computer
there should be a way to reinit the gpu:
  "In windows go to device manager -> display adapters, click on card which you want reset, right click, 
  disable hardware and with keyboard shortcuts (CTRL+ ....) without screen again enable card - card is reseted..."
  press Win+Ctrl+Shift+B.


*/

