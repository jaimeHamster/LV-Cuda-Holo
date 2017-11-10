/*
  Title     : custom.h
  Project   : GPU Analysis Toolkit 2012
  Platforms : Windows x86/x64
  Purpose   : Code Sample - Calling Custom GPU Functions
  */



#ifndef CUDADLL_H
#define CUDADLL_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <windows.h>
#include <string.h>
#include <ctype.h>
#include <math.h>


//declare all the functions I want to include in the .dll HERE
#ifdef __cplusplus
extern "C" {
#endif
//////////////////////////
////////////
//Main functions that operate on holograms
/////////////



__declspec(dllexport) void __cdecl PropagateZslices(float* h_bfpMag, float* h_bfpPhase,
		float* h_ImgOutRe, float* h_ImgOutIm,
		float* zscale, int* arraySize, float* imgProperties);


// non-optimised version of the processing
__declspec(dllexport) void __cdecl Propagate3Dz_CSG(float* h_KernelRE, float* h_KernelIm,
	float* h_bfpRe, float* h_bfpIm,
	float* h_ImgOutRe, float* h_ImgOutIm,
	float* zscale, int* arraySize);

////////////////////////////////////
// useful Benchmark functions
__declspec(dllexport) void __cdecl BottleNeck3DTransform_CSG(float* h_KernelRE, float* h_KernelIm,
	float* h_bfpRe, float* h_bfpIm,
	float* h_KernelModRe, float* h_KernelModIm,
	float* zscale, int* arraySize);

__declspec(dllexport) void __cdecl Make3DTransform_CSG(float* h_KernelRE, float* h_KernelIm,
	float* h_bfpRe, float* h_bfpIm,
	float* h_KernelModRe, float* h_KernelModIm,
	float* zscale, int* arraySize);


///////
// Simple examples on how to run GPU Kernels

__declspec(dllexport) void __cdecl myGPUfunction(const char *s);

__declspec(dllexport) void __cdecl scaleArrayGPU(float* h_a, float alpha, int arraySize);

__declspec(dllexport) void __cdecl scaleArray2DGPU(float* h_2a, float alpha, int* array2DSize);

__declspec(dllexport) void __cdecl scaleArrayGPU_CSG(float* h_KernelRE, float* h_KernelIm,
	float* h_KernelModRe, float* h_KernelModIm,
	float ascale, int* arraySize);

__declspec(dllexport) void __cdecl generate3DKernel_CSG(float* h_KernelRE, float* h_KernelIm,
	float* h_KernelModRe, float* h_KernelModIm,
	float* zscale, int* arraySize);


#ifdef __cplusplus
}
#endif

#endif  // CUDADLL_H