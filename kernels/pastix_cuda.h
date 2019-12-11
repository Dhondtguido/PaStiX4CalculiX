/**
 * @file pastix_cuda.h
 *
 * PaStiX GPU kernel header.
 *
 * @copyright 2016-2018 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @version 6.0.1
 * @author Mathieu Faverge
 * @date 2018-07-16
 *
 */
#ifndef _pastix_cuda_h_
#define _pastix_cuda_h_

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_BATCH_COUNT 16

typedef struct gemm_param_s{
    const void *Aptr;
    void *Cptr;
    pastix_int_t M;
    pastix_int_t lda;
    pastix_int_t ldc;
} gemm_param_t;

typedef struct gemm_params_s {
    gemm_param_t p[MAX_BATCH_COUNT];
} gemm_params_t;

void
pastix_zgemm_vbatched_nt(
    pastix_trans_t transB,
    pastix_int_t n, pastix_int_t k,
    cuDoubleComplex alpha,
    const cuDoubleComplex * dB, pastix_int_t lddb,
    cuDoubleComplex beta,
    pastix_int_t max_m, pastix_int_t batchCount, cudaStream_t stream,
    gemm_params_t params );

void
pastix_cgemm_vbatched_nt(
    pastix_trans_t transB,
    pastix_int_t n, pastix_int_t k,
    cuFloatComplex alpha,
    const cuFloatComplex * dB, pastix_int_t lddb,
    cuFloatComplex beta,
    pastix_int_t max_m, pastix_int_t batchCount, cudaStream_t stream,
    gemm_params_t params );

void
pastix_dgemm_vbatched_nt(
    pastix_trans_t transB,
    pastix_int_t n, pastix_int_t k,
    double alpha,
    const double * dB, pastix_int_t lddb,
    double beta,
    pastix_int_t max_m, pastix_int_t batchCount, cudaStream_t stream,
    gemm_params_t params );

void
pastix_sgemm_vbatched_nt(
    pastix_trans_t transB,
    pastix_int_t n, pastix_int_t k,
    float alpha,
    const float * dB, pastix_int_t lddb,
    float beta,
    pastix_int_t max_m, pastix_int_t batchCount, cudaStream_t stream,
    gemm_params_t params );


void
pastix_fermi_zgemmsp(
    char TRANSA, char TRANSB, int m , int n , int k ,
    cuDoubleComplex alpha, const cuDoubleComplex *d_A, int lda,
                           const cuDoubleComplex *d_B, int ldb,
    cuDoubleComplex beta,        cuDoubleComplex *d_C, int ldc,
    int blocknbr, const int *blocktab, int fblocknbr, const int *fblocktab,
    cudaStream_t stream );

void
pastix_fermi_cgemmsp(
    char TRANSA, char TRANSB, int m , int n , int k ,
    cuFloatComplex alpha, const cuFloatComplex *d_A, int lda,
                          const cuFloatComplex *d_B, int ldb,
    cuFloatComplex beta,        cuFloatComplex *d_C, int ldc,
    int blocknbr, const int *blocktab, int fblocknbr, const int *fblocktab,
    cudaStream_t stream );

void
pastix_fermi_dgemmsp(
    char TRANSA, char TRANSB, int m , int n , int k ,
    double alpha, const double *d_A, int lda,
                  const double *d_B, int ldb,
    double beta,        double *d_C, int ldc,
    int blocknbr, const int *blocktab, int fblocknbr, const int *fblocktab,
    cudaStream_t stream );

void
pastix_fermi_sgemmsp(
    char TRANSA, char TRANSB, int m , int n , int k ,
    float alpha, const float *d_A, int lda,
                 const float *d_B, int ldb,
    float beta,        float *d_C, int ldc,
    int blocknbr, const int *blocktab, int fblocknbr, const int *fblocktab,
    cudaStream_t stream );
    
    
                        
void pastix_z_spmv_one_base(
			pastix_int_t m,
			pastix_complex64_t alpha,
			const pastix_complex64_t* dval,
			pastix_int_t* drowptr,
			pastix_int_t* dcolind,
			const pastix_complex64_t* dx,
			pastix_complex64_t beta,
			pastix_complex64_t* dy );
                        
void pastix_c_spmv_one_base(
			pastix_int_t m,
			pastix_complex32_t alpha,
			const pastix_complex32_t* dval,
			pastix_int_t* drowptr,
			pastix_int_t* dcolind,
			const pastix_complex32_t* dx,
			pastix_complex32_t beta,
			pastix_complex32_t* dy );
                        
void pastix_d_spmv_one_base(
			pastix_int_t m,
			double alpha,
			const double* dval,
			pastix_int_t* drowptr,
			pastix_int_t* dcolind,
			const double* dx,
			double beta,
			double* dy);
                        
void pastix_s_spmv_one_base(
			pastix_int_t m,
			float alpha,
			const float* dval,
			pastix_int_t* drowptr,
			pastix_int_t* dcolind,
			const float* dx,
			float beta,
			float* dy);
                        
void pastix_z_spmv(
			pastix_int_t m,
			pastix_complex64_t alpha,
			const pastix_complex64_t* dval,
			pastix_int_t* drowptr,
			pastix_int_t* dcolind,
			const pastix_complex64_t* dx,
			pastix_complex64_t beta,
			pastix_complex64_t* dy,
			cudaStream_t stream );
                        
void pastix_c_spmv(
			pastix_int_t m,
			pastix_complex32_t alpha,
			const pastix_complex32_t* dval,
			pastix_int_t* drowptr,
			pastix_int_t* dcolind,
			const pastix_complex32_t* dx,
			pastix_complex32_t beta,
			pastix_complex32_t* dy,
			cudaStream_t stream );
                        
void pastix_d_spmv(
			pastix_int_t m,
			double alpha,
			const double* dval,
			pastix_int_t* drowptr,
			pastix_int_t* dcolind,
			const double* dx,
			double beta,
			double* dy,
			cudaStream_t stream );
                        
void pastix_s_spmv(
			pastix_int_t m,
			float alpha,
			const float* dval,
			pastix_int_t* drowptr,
			pastix_int_t* dcolind,
			const float* dx,
			float beta,
			float* dy,
			cudaStream_t stream );
    
void pastix_z_spmv_perm(
			pastix_int_t m,
			pastix_complex64_t alpha,
			const pastix_complex64_t* dval,
			pastix_int_t* drowptr,
			pastix_int_t* dcolind,
			const pastix_complex64_t* dx,
			pastix_complex64_t beta,
			pastix_complex64_t* dy,
			pastix_int_t* perm );
    
void pastix_c_spmv_perm(
			pastix_int_t m,
			pastix_complex32_t alpha,
			const pastix_complex32_t* dval,
			pastix_int_t* drowptr,
			pastix_int_t* dcolind,
			const pastix_complex32_t* dx,
			pastix_complex32_t beta,
			pastix_complex32_t* dy,
			pastix_int_t* perm);
    
void pastix_d_spmv_perm(
			pastix_int_t m,
			double alpha,
			const double* dval,
			pastix_int_t* drowptr,
			pastix_int_t* dcolind,
			const double* dx,
			double beta,
			double* dy,
			pastix_int_t* perm );
    
void pastix_s_spmv_perm(
			pastix_int_t m,
			float alpha,
			const float* dval,
			pastix_int_t* drowptr,
			pastix_int_t* dcolind,
			const float* dx,
			float beta,
			float* dy,
			pastix_int_t* perm);
			

#ifdef __cplusplus
}
#endif


#endif /* _pastix_cuda_h_ */
