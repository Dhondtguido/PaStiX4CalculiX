/**
 *
 * @file gpu_z_spmv.c
 *
 * @copyright 2012-2018 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * PaStiX GPU SPMV
 *
 * @version 6.0.1
 * @author Mathieu Faverge
 * @author Pierre Ramet
 * @author Xavier Lacoste
 * @date 2018-07-16
 * @precisions normal z -> c d s
 *
 **/

#include "gpu_z_spmv.h"
#include "pastix_cuda.h"

void
gpu_z_spmv(		pastix_int_t n, 
				pastix_complex64_t alpha,
				pastix_complex64_t beta,
		  const pastix_complex64_t *A,
		  const pastix_complex64_t *x,
				pastix_complex64_t *y,
				pastix_int_t* rowptr,
				pastix_int_t* colind ){
	printf("GPU ZSPMV\n");
	pastix_z_spmv_one_base(n, alpha, A, rowptr, colind, x, beta, y );
}


void
gpu_z_spmv_perm(pastix_int_t n, 
				pastix_complex64_t alpha,
				pastix_complex64_t beta,
		  const pastix_complex64_t *A,
		  const pastix_complex64_t *x,
				pastix_complex64_t *y,
				pastix_int_t* rowptr,
				pastix_int_t* colind,
                pastix_int_t* perm ){
	printf("GPU ZSPMV PERM\n");
	pastix_z_spmv_perm(n, alpha, A, rowptr, colind, x, beta, y, perm);
}
