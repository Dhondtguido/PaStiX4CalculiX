#include "SpMV.h"
#include "stdio.h"

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

SpMVDoubleWarp* spmv;

EXTERNC void createLightSpMV(int64_t m, int64_t nnz){
	Options* opt = new Options;
	opt->_numRows = m;
	opt->_numCols = m;
	opt->_numValues = nnz;
	if(!opt->getGPUs())
		printf("Error while scanning for GPU\n");
	
	spmv = new SpMVDoubleWarp(opt);
	cudaDeviceSetCacheConfig (cudaFuncCachePreferL1);
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
	delete spmv;
}

#undef EXTERNC
