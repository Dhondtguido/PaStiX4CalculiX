#include "SpMV.h"
#include "stdio.h"
/*
#include <cuda_runtime.h>  // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>      // cusparseSpMV
#include <stdio.h>         // printf
#include <stdlib.h>        // EXIT_FAILURE

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}*/

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif
/*
int64_t glob_m = 0;
cusparseHandle_t     handle = 0;
cusparseSpMatDescr_t matA;
cusparseDnVecDescr_t vecX, vecY;
void*  dBuffer    = NULL;
*/
SpMVDoubleWarp* spmv;

EXTERNC void createLightSpMV(int64_t m, int64_t nnz, int64_t* drowptr, int64_t* dcolind, double* dvalues){ 
/*	glob_m = m;

    // CUSPARSE APIs
    size_t bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, m, m, nnz,
                                      drowptr, dcolind, dvalues,
                                      CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                                      CUSPARSE_INDEX_BASE_ONE, CUDA_R_64F) )
    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, m, NULL, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, m, NULL, CUDA_R_64F) )
    // Create dense vector y
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                 CUSPARSE_MV_ALG_DEFAULT, &bufferSize) )
    CHECK_CUSPARSE( cudaMalloc(&dBuffer, bufferSize) )
*/
	
	
	Options* opt = new Options;
	opt->_numRows = m;
	opt->_numCols = m;
	opt->_numValues = nnz;
	if(!opt->getGPUs())
		printf("Error while scanning for GPU\n");
	
	spmv = new SpMVDoubleWarp(opt);
	//cudaDeviceSetCacheConfig (cudaFuncCachePreferL1);
	spmv->loadData();
}

EXTERNC void performLightLsMV(
    double alpha,
    double* dval,
    int64_t* drowptr,
    int64_t* dcolind,
    double* dx,
    double beta,
    double* dy){
		
/*
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, m, dx, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, m, dy, CUDA_R_64F) )
		
		
    // execute SpMV
    CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                 CUSPARSE_MV_ALG_DEFAULT, dBuffer) )
*/
	
	spmv->_rowOffsets[0] = drowptr;
	spmv->_colIndexValues[0] = dcolind;
	spmv->_numericalValues[0] = dval;
	spmv->_vectorX[0] = dx;
	spmv->_vectorY[0] = dy;
	
	spmv->_alpha = alpha;
	spmv->_beta = beta;
	
	spmv->spmvKernel();
}

EXTERNC void destroyLightSpMV(){
	/*
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    
    CHECK_CUDA( cudaFree(dBuffer) )
	*/
	
	
	spmv->_rowOffsets[0] = NULL;
	spmv->_colIndexValues[0] = NULL;
	spmv->_numericalValues[0] = NULL;
	delete spmv;
}

#undef EXTERNC
