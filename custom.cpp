/*
  Title     : custom.cpp
  Project   : GPU Analysis Toolkit 2012
  Platforms : Windows x86/x64
  Purpose   : Code Sample - Calling Custom GPU Functions

  FILE DESCRIPTION
  ================
  This code sample defines functions for use with data wrappers constructed
  based on the LVGPU SDK.

  Usage
  -----
  Add all required items and any optional items to the file containing the source 
  of the custom GPU function, modify the contents as needed and rebuild the library.

  If access to the GPU function source is not available, then this source includes 
  the skeleton for building a stand-alone library.

                       ******** IMPORTANT NOTICE ********
  The automated cleanup mechanism may operate after the LabVIEW application closes. This
  means that any libraries loaded by LabVIEW for that application instance may be 
  unloaded. When a library containing a cleanup procedure is unloaded, the address 
  to that function stored in the SDK data wrapper is no longer valid and calling the
  function via its address generates exceptions/errors.

  Holding a reference to the library avoids this problem but the methods to do so are
  beyond the scope of the content presented here.
                       **********************************

  Build Configuration
  -------------------
  The following include paths are required when using lvgpu.h:
    [LVDIR]\resource\lvgpu  - location of lvgpu.h
    .\dep                   - location of lvgpu.h dependencies (ni_extcode)
    [LVDIR]\cintools        - location of ni_extcode dependencies
 
  The current project is configured to build a 32-bit DLL with and w/out debug 
  information. After a build, the binary is copied to the bin directory even if it
  contains debug information.
  
  Required items
  --------------
  includes:
  custom.h - contains lvgpu.h defining LVGPU SDK types and compiles the cleanup
             procedure components appropriately (exported, calling convention, etc).

  function definitions:
  GPUDeviceData#CleanupProc - defines the function responsible for free the device
                              reference. This interface must conform to the LVGPU
                              SDK's cleanup procedure interface which has three inputs:

                              _arrayData   : a reference to host data (rarely used)
                              _customData  : a reference to device data
                              _cleanupData : an internal structure used to protect
                                             array and custom data. In the case of
                                             the SDK wrapper, this structure 
                                             references state information

  GetGPUDeviceData#CleanupProcAddress - returns the address of the cleanup procedure
                                        for the loaded library.

  Optional items
  --------------
  function definitions:
  GPUDeviceData1_GetInstance - sample function showing how to retrieve the device
                               reference stored inside the cleanup data structure

  GPUDeviceData1_Apply - wrapper function for using the state information to prepare
                         for cleanup.

  GPUDeviceData#_Free -  wrapper functions for releasing the specific device reference.

  DllMain - function required to build a Windows-based shared library
*/

#include "windows.h"
#include "custom.h"

/*******************************/
/* support macros - C vs C++   */
/*******************************/
#if defined(__cplusplus)
#define DATA_CAST(__TYPE__, __REF__) static_cast<__TYPE__>(__REF__)
#define FCT_CAST(__TYPE__, __REF__)  reinterpret_cast<__TYPE__>(__REF__)
#else
#define DATA_CAST(__TYPE__, __REF__) (__TYPE__)(__REF__)
#define FCT_CAST(__TYPE__, __REF__) DATA_CAST(__TYPE__, __REF__)
#endif

/*********************************/
/* private functions definitions */
/*********************************/
static BOOL GPUDeviceData1_GetInstance(GPUCleanupData_t _cleanupData, GPUDeviceData1_t * _data)
{
   if (_cleanupData && _data) {
      *_data = DATA_CAST(GPUDeviceData1_t, _cleanupData->customData);
      return TRUE;
   }
   return FALSE;
}

static BOOL GPUDeviceData1_Apply(GPUDeviceData1_t /* _data */) {
   /* Apply '_data' here and return apply status */
   return TRUE;
}

static void GPUDeviceData1_Free(void * /* _ref */) {
   /* Free the device reference stored in '_ref' */
}

static void GPUDeviceData2_Free(void * /* _ref */) {
   /* Free the device reference stored in '_ref' */
}

/*******************************/
/* public function definitions */
/*******************************/
/*
   GPUDeviceData1CleanupProc

   Description:
      Frees the device reference stored using an SDK wrapper of GPUDeviceData1_t type.

   Inputs:
      _ref - the device reference of type GPUDeviceData1_t

   Outputs:
      <none>

   Comments:
      This function is exported from the library even though it
      is not called explicitly. It is invoked indirectly by an address stored
      inside the SDK data wrapper when the wrapper is deleted.

      In the G portion of the SDK wrapper, the GPUDeviceData1_t reference is stored using 
      a type definition -- GPUDeviceData1.ctl.
*/
void GPUDeviceData1CleanupProc(void * /* unused */, 
                               void * _ref, 
                               GPUCleanupData_t /* unused */)
{
   GPUDeviceData1_Free(_ref);
   return;
}

/*
   GPUDeviceData2CleanupProc

   Description:
      Frees the device reference stored using an SDK wrapper of GPUDeviceData2_t type.

   Inputs:
      _ref         - the device reference of type GPUDeviceData2_t
      _cleanupData - the additional state information

   Outputs:
      <none>

   Comments:
      This function is exported from the library even though it
      is not called explicitly. It is invoked indirectly by an address stored
      inside the SDK data wrapper when the wrapper is deleted.

      In the G portion of the SDK wrapper, the GPUDeviceData2_t reference is stored using 
      a type definition -- GPUDeviceData2.ctl.

      The state information is stored using another SDK data wrapper - one designed to
      store a reference of type GPUDeviceData1_t. The internal structure used to protect
      this data is of type GPUCleanupData_t. To see how to retrieve the reference to the 
      state information, see the definition of GPUDeviceData1_GetInstance().
*/
void GPUDeviceData2CleanupProc(void * /* unused */, 
                               void * _ref, 
                               GPUCleanupData_t _cleanupData)
{
   GPUDeviceData1_t data;
   if (GPUDeviceData1_GetInstance(_cleanupData, &data)) {
      if (GPUDeviceData1_Apply(data)) {
         GPUDeviceData2_Free(_ref);
      }
   }
   return;
}


/*
   GetGPUDeviceData1CleanupProcAddress
   GetGPUDeviceData2CleanupProcAddress

   Description:
      Returns the address of the cleanup procedure in the loaded library.

   Outputs:
      uintptr_t - the address stored in an unsigned pointer-sized integer

   Comments:
      This function is called from the Get Cleanup Procedure Address.vi in
      the SDK data class.
*/
uintptr_t GetGPUDeviceData1CleanupProcAddress()
{
      GPUDataCallbackFuncPtr_t fct = GPUDeviceData1CleanupProc;
      return FCT_CAST(uintptr_t, fct);
}

uintptr_t GetGPUDeviceData2CleanupProcAddress()
{
      GPUDataCallbackFuncPtr_t fct = GPUDeviceData2CleanupProc;
      return FCT_CAST(uintptr_t, fct);
}


//============================================================================
//  Library Management
//============================================================================
extern "C" BOOL APIENTRY 
DllMain(HANDLE /*hModule*/, DWORD ul_reason_for_call, LPVOID /*lpReserved*/)
{
   BOOL  retVal = TRUE;

	switch (ul_reason_for_call) {
   	case DLL_PROCESS_ATTACH:
			break;
		case DLL_THREAD_ATTACH:
		case DLL_THREAD_DETACH:
   		break;
		case DLL_PROCESS_DETACH:
			break;
	}
	return retVal;
}