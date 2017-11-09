//////////////////
// Temporarily empty file
///////////////////

//// Working place to make sure I can manipulate LV 1D complex data structures. 
#include "extcode.h"
#include <lv_prolog.h>
#include <lv_epilog.h> //set up the correct alignment for LabVIEW data
#include <math.h>
#include <cufft.h>
#include "CudaDLL.h"


/* LabVIEW created typedefs */
//input complex 2D data:
typedef struct {
	int32_t dimSizes[2]; //2D array dimension size
	cmplx64 idata[1]; // data is here
} LVCmplxArray1; // define LV 2D data complex as an input handle!
typedef LVCmplxArray1 **In2DCmplxHdl;

typedef struct {
	int32_t dimSizes[2];
	cmplx64 odata[1];
} LVCmplxArray2;
typedef LVCmplxArray2 **Out2DCmplxHdl;

void testFunction( In2DCmplxHdl dataIn, Out2DCmplxHdl dataOut){
	//How to extract the basic data and multiply it!
	int indx;
	//(*dataOut)->odata[indx]=(*dataIn)->idata[indx];

	size_t memsize= (*dataIn)->dimSizes[0]* (*dataIn)->dimSizes[1]*sizeof(cufftComplex);
	
	cufftComplex *d_data;
//	d_data = (*dataIn)->idata; //Cannot copy cmplx64 to cufftComplex... they are incompatible!! :(
//	cudaMemcpy(d_data, (*dataIn)->idata, memsize, cudaMemcpyHostToDevice);

}