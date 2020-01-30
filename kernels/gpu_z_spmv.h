/**
 *
 * @file gpu_z_spmv.h
 *
 * @copyright 2012-2018 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 *
 * @version 6.0.1
 * @author Mathieu Faverge
 * @author Pierre Ramet
 * @author Xavier Lacoste
 * @date 2018-07-16
 * @precisions normal z -> c d s
 *
 **/

#include "pastix.h"
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
                        
void
gpu_z_spmv(		pastix_int_t n, 
				pastix_complex64_t alpha,
				pastix_complex64_t beta,
		        const pastix_complex64_t *A,
		        const pastix_complex64_t *x,
				pastix_complex64_t *y,
				pastix_int_t* rowptr,
				pastix_int_t* colind );
