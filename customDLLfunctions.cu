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

__device__ static __inline__ float cmagf(float x, float y)
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

__global__ void ConvertCmplx2Polar(float* inRe, float* inIm, float* mag, float* phase, int size) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = threadID; i < size; i += numThreads)
	{
		phase[i] = atan2f(inIm[i], inRe[i]);
		mag[i] = cmagf(inIm[i], inRe[i]);
	}
}


// Create a 3D complex array from multiplication of a 2D array by 1D array
//could be some form of slow down in the code if it is done with integer division / modulo operations
//although it seems it is the only easy way to keep the threads parallelizable

__global__ void Create3DKernel(cufftComplex* KernelOut, cufftComplex* KernelIn, float* zrange, int size, int imgsize, int zsize)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	//its imperative that operations are performed on the individual elements,can not appyling the scaling simultaneously .x/.y
	for (int i = threadID; i < size; i += numThreads)
	{
		KernelOut[i].x = zrange[i%zsize]*KernelIn[i%imgsize].x;
		KernelOut[i].y = zrange[i%zsize]*KernelIn[i%imgsize].y;
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
__global__ void Create3DTransferFunction(cufftComplex* KernelOut, cufftComplex* KernelIn, cufftComplex* bfpIn, float* zscale, int size, int imgsize, int zsize)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	//additional counters
	for (int i = threadID ; i < size; i += numThreads)
	{
		int j = i / imgsize; //doing modulo zsize gives weird results!
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


void BottleNeck3DTransform_CSG(float* h_KernelRE, float* h_KernelIm,
	float* h_bfpRe, float* h_bfpIm,
	float* h_KernelModRe, float* h_KernelModIm,
	float* zscale, int* arraySize) {

	//Extract the size of the 2D and 3D arrays, and their respect allocation sizes
	int row = arraySize[0];
	int column = arraySize[1];
	int zrange = arraySize[2];
	int numElements = row*column;
	int size3Darray = row*column*zrange;

	const int BlockSize = 512;
	int GridSize = 32 * 16* 4;// (size3Darray + BlockSize - 1) / BlockSize;

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
	float alpha = 0.4;
	ScaleArray<< <GridSize, BlockSize >> > (d_zscale, alpha, zrange);
	//Create3DKernel <<<GridSize, BlockSize >> > (d_3DKernel, d_Kernel, d_zscale, size3Darray, numElements,zrange);

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

	const int BlockSize = 512;
	int GridSize = 32*16*4;

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
	Create3DTransferFunction <<<GridSize, BlockSize >>> (d_3DKernel, d_Kernel, d_bfpIn, d_zscale, size3Darray, numElements, zrange);

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
	
	const int BlockSize = 512;
	int GridSize = 32*16*4;

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
	Create3DTransferFunction <<<GridSize, BlockSize >>> (d_3DKernel, d_Kernel, d_bfpIn, d_zscale, size3Darray, numElements,zrange);


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

	/* ////////// Execute the transform out-of-place */
	/*cufftComplex *d_3Dimg;
	cudaMalloc((void**)&d_3Dimg, mem3dsize);
	
	if (cufftExecC2C(BatchFFTPlan, d_3DKernel, d_3Dimg, CUFFT_INVERSE) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT Error: Failed to execute plan\n");
		return;
	}
	*/

	//////// Execute the transform in-place
	if (cufftExecC2C(BatchFFTPlan, d_3DKernel, d_3DKernel, CUFFT_INVERSE) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT Error: Failed to execute plan\n");
		return;
	}


	/*if (cudaDeviceSynchronize() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
		return;
	} */

	
	//However, unlike a normal sequential program on your host (The CPU) will continue to execute the next lines of code in your program.
	//cudaDeviceSynchronize makes the host (The CPU) wait until the device (The GPU) have finished executing ALL the threads you have started,
	//and thus your program will continue as if it was a normal sequential program.

	//free data
	cufftDestroy(BatchFFTPlan);
	//cudaFree(idata);
	//cudaFree(odata);


	
	////////
	// FFT ends
	///////



	//Copy device memory to host
	cufftComplex *h_3Dimg;
	h_3Dimg = new cufftComplex[size3Darray];
	cudaMemcpy(h_3Dimg, d_3DKernel, mem3dsize, cudaMemcpyDeviceToHost);

	//deallocate CUDA memory
	cudaFree(d_bfpIn);
	cudaFree(d_Kernel);
	cudaFree(d_3DKernel);
	cudaFree(d_zscale);
	//cudaFree(d_3Dimg);

	//transfer the complex2d array back as a processed individual re and imaginary 2d arrays
	// i think it is prudent that I allocate the 2D space in the dll program
	for (int i = 0; i < size3Darray; i++) {
		h_ImgOutRe[i] = h_3Dimg[i].x;
		h_ImgOutIm[i] = h_3Dimg[i].y;
	}

	//deallocate Host memory
	delete[] h_Kernel;
	delete[] h_3Dimg;

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
//	int zrange = arraySize[2];
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
	Create3DKernel << <32, 256 >> > (d_3DKernel, d_Kernel, d_zscale, size3Darray, numElements,zrange);

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

