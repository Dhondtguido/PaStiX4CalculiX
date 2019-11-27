#include <cuda_fp16.h>
#include <cstdlib>
#include <cstdio>
#include "half_prec_utility.h"
#include <cublas_v2.h>

__global__
void downcast_block_kernel(const float* src_block, int width, int height, int ld, half* dest_block){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int row = i % height;
	int col = i / height;
	//#pragma unroll
	if(i < width * height){
        dest_block[row + col * ld] = __float2half(src_block[row + col * ld]);
	}
}

__global__
void upcast_block_kernel(__half* src_block, int width, int height, int ld, float* dest_block){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int row = i % height;
	int col = i / height;
	//#pragma unroll
	if(i < width * height){
        dest_block[row + col * ld] = __half2float(src_block[row + col * ld]);
	}
}


__host__
void downcast_block(const void* src_block, int width, int height, int ld, void* dest_block){
    __half* hp_block = (__half*) dest_block;    
	const float* sp_block = (float*) src_block;
    downcast_block_kernel<<<(width*height+511)/512,512>>>(sp_block, width, height, ld, hp_block);
}

__host__
void upcast_block(void* src_block, int width, int height, int ld, void* dest_block){
    __half* hp_block = (__half*) src_block;    
	float* sp_block = (float*) dest_block;
    upcast_block_kernel<<<(width*height+511)/512,512>>>(hp_block, width, height, ld, sp_block);
}

__host__
void wrapHgemm (cublasHandle_t* handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *A, int lda, const void *B, int ldb, void *C, int ldc){
	float mzone = -1;
	float zone = 1;
	__half hpmzone = __float2half(mzone);
	__half hpzone = __float2half(zone);

    cublasHgemm(*handle, transa, transb, m, n, k, &hpmzone, (__half*)A, lda, (__half*)B, ldb, &hpzone, (__half*)C, ldc);
}
