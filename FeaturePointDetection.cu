//want to include the feature point detection using templates


//Calculate the sum image, cumsum over all pixels
#include "CudaDLL.h"
#include <stdio.h>
#include <cufft.h>
#include <cuComplex.h>
#include <device_functions.h>
#include <math.h>
#include <float.h>


__global__ void transposeNaive(float *odata, const float *idata)
{
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		odata[x*width + (y + j)] = idata[(y + j)*width + x];
}



// coalesced transpose
// Uses shared memory to achieve coalesing in both reads and writes
// Tile width == #banks causes shared memory bank conflicts.
__global__ void transposeCoalesced(float *odata, const float *idata)
{
	__shared__ float tile[TILE_DIM][TILE_DIM];

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		tile[threadIdx.y + j][threadIdx.x] = idata[(y + j)*width + x];

	__syncthreads();

	x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		odata[(y + j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}


// No bank-conflict transpose
// Same as transposeCoalesced except the first tile dimension is padded 
// to avoid shared memory bank conflicts.
__global__ void transposeNoBankConflicts(float *odata, const float *idata)
{
	__shared__ float tile[TILE_DIM][TILE_DIM + 1];

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		tile[threadIdx.y + j][threadIdx.x] = idata[(y + j)*width + x];

	__syncthreads();

	x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		odata[(y + j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}


__global__ void KernTranspose(float* img_out, float* img_in, int row, int column)
{
	__shared__ Tin temp[BLOCK_DIM][BLOCK_DIM + 1];
	int xIndex = blockIdx.x*BLOCK_DIM + threadIdx.x;
	int yIndex = blockIdx.y*BLOCK_DIM + threadIdx.y;

	if ((xIndex < row) && (yIndex < column)) {
		temp[threadIdx.y][threadIdx.x] = in(xIndex, yIndex);
	}

	__syncthreads();

	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;

	if ((xIndex < in.h) && (yIndex < in.w)) {
		out(xIndex, yIndex) = temp[threadIdx.x][threadIdx.y];
	}
}

__global__ void SumImage(float* img2Darray, float* sumimg, int row, int column)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	const int imgsize = row * column;

	//additional counters
	for (int i = threadID; i < imgsize; i += numThreads)
	{
		int idx = k % row;
		int idy = k / row;
		




	}
}




int main(int argc, char **argv)
{
	const int nx = 1024;
	const int ny = 1024;
	const int mem_size = nx * ny * sizeof(float);

	dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM, 1);
	dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

	int devId = 0;
	if (argc > 1) devId = atoi(argv[1]);

	cudaDeviceProp prop;
	checkCuda(cudaGetDeviceProperties(&prop, devId));
	printf("\nDevice : %s\n", prop.name);
	printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n",
		nx, ny, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
	printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
		dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

	checkCuda(cudaSetDevice(devId));

	float *h_idata = (float*)malloc(mem_size);
	float *h_cdata = (float*)malloc(mem_size);
	float *h_tdata = (float*)malloc(mem_size);
	float *gold = (float*)malloc(mem_size);

	float *d_idata, *d_cdata, *d_tdata;
	checkCuda(cudaMalloc(&d_idata, mem_size));
	checkCuda(cudaMalloc(&d_cdata, mem_size));
	checkCuda(cudaMalloc(&d_tdata, mem_size));

	// check parameters and calculate execution configuration
	if (nx % TILE_DIM || ny % TILE_DIM) {
		printf("nx and ny must be a multiple of TILE_DIM\n");
		goto error_exit;
	}

	if (TILE_DIM % BLOCK_ROWS) {
		printf("TILE_DIM must be a multiple of BLOCK_ROWS\n");
		goto error_exit;
	}

	// host
	for (int j = 0; j < ny; j++)
		for (int i = 0; i < nx; i++)
			h_idata[j*nx + i] = j * nx + i;

	// correct result for error checking
	for (int j = 0; j < ny; j++)
		for (int i = 0; i < nx; i++)
			gold[j*nx + i] = h_idata[i*nx + j];

	// device
	checkCuda(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

	// events for timing
	cudaEvent_t startEvent, stopEvent;
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));
	float ms;

	// ------------
	// time kernels
	// ------------
	printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");

	// ----
	// copy 
	// ----
	printf("%25s", "copy");
	checkCuda(cudaMemset(d_cdata, 0, mem_size));
	// warm up
	copy << <dimGrid, dimBlock >> > (d_cdata, d_idata);
	checkCuda(cudaEventRecord(startEvent, 0));
	for (int i = 0; i < NUM_REPS; i++)
		copy << <dimGrid, dimBlock >> > (d_cdata, d_idata);
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	checkCuda(cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost));
	postprocess(h_idata, h_cdata, nx*ny, ms);

	// -------------
	// copySharedMem 
	// -------------
	printf("%25s", "shared memory copy");
	checkCuda(cudaMemset(d_cdata, 0, mem_size));
	// warm up
	copySharedMem << <dimGrid, dimBlock >> > (d_cdata, d_idata);
	checkCuda(cudaEventRecord(startEvent, 0));
	for (int i = 0; i < NUM_REPS; i++)
		copySharedMem << <dimGrid, dimBlock >> > (d_cdata, d_idata);
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	checkCuda(cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost));
	postprocess(h_idata, h_cdata, nx * ny, ms);

	// --------------
	// transposeNaive 
	// --------------
	printf("%25s", "naive transpose");
	checkCuda(cudaMemset(d_tdata, 0, mem_size));
	// warmup
	transposeNaive << <dimGrid, dimBlock >> > (d_tdata, d_idata);
	checkCuda(cudaEventRecord(startEvent, 0));
	for (int i = 0; i < NUM_REPS; i++)
		transposeNaive << <dimGrid, dimBlock >> > (d_tdata, d_idata);
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	checkCuda(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
	postprocess(gold, h_tdata, nx * ny, ms);

	// ------------------
	// transposeCoalesced 
	// ------------------
	printf("%25s", "coalesced transpose");
	checkCuda(cudaMemset(d_tdata, 0, mem_size));
	// warmup
	transposeCoalesced << <dimGrid, dimBlock >> > (d_tdata, d_idata);
	checkCuda(cudaEventRecord(startEvent, 0));
	for (int i = 0; i < NUM_REPS; i++)
		transposeCoalesced << <dimGrid, dimBlock >> > (d_tdata, d_idata);
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	checkCuda(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
	postprocess(gold, h_tdata, nx * ny, ms);

	// ------------------------
	// transposeNoBankConflicts
	// ------------------------
	printf("%25s", "conflict-free transpose");
	checkCuda(cudaMemset(d_tdata, 0, mem_size));
	// warmup
	transposeNoBankConflicts << <dimGrid, dimBlock >> > (d_tdata, d_idata);
	checkCuda(cudaEventRecord(startEvent, 0));
	for (int i = 0; i < NUM_REPS; i++)
		transposeNoBankConflicts << <dimGrid, dimBlock >> > (d_tdata, d_idata);
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	checkCuda(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
	postprocess(gold, h_tdata, nx * ny, ms);

error_exit:
	// cleanup
	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
	checkCuda(cudaFree(d_tdata));
	checkCuda(cudaFree(d_cdata));
	checkCuda(cudaFree(d_idata));
	free(h_idata);
	free(h_tdata);
	free(h_cdata);
	free(gold);
}

