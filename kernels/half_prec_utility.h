#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

#include <cublas_v2.h>
 
#ifndef _half_prec_utility_h_
#define _half_prec_utility_h_

EXTERNC void downcast_block(const void* src_block, int width, int height, int ld, void* dest_block, cudaStream_t* stream);
EXTERNC void upcast_block( void* src_block, int width, int height, int ld, void* dest_block, cudaStream_t* stream);

EXTERNC void wrapHgemm (cublasHandle_t* handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *A, int lda, const void *B, int ldb, void *C, int ldc);
#endif /* _half_prec_utility_h_ */

#undef EXTERNC
