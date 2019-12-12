/**
 *
 * @file cpu_z_spmv.c
 *
 * @copyright 2012-2018 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * PaStiX CPU SPMV
 *
 * @version 6.0.1
 * @author Mathieu Faverge
 * @author Pierre Ramet
 * @author Xavier Lacoste
 * @date 2018-07-16
 * @precisions normal z -> c d s
 *
 **/

#include "cpu_z_spmv.h"


void
cpu_z_spmv(		pastix_int_t n, 
				pastix_complex64_t alpha,
				pastix_complex64_t beta,
		  const pastix_complex64_t *A,
		  const pastix_complex64_t *x,
				pastix_complex64_t *y,
				pastix_int_t* rowptr,
				pastix_int_t* colind){
	
	#pragma omp parallel for
    for( pastix_int_t i=0; i<n; i++)
    {
		pastix_complex64_t dot = 0.0;
		
        for( pastix_int_t j=rowptr[i]-1; j<rowptr[i+1]-1; j++)
        {
            dot += A[j] * x[ colind[j] -1 ];
        }
        
		y[i] = alpha * dot + beta * y[i];
    }
}

void
cpu_z_spmv_perm(pastix_int_t n, 
				pastix_complex64_t alpha,
				pastix_complex64_t beta,
		  const pastix_complex64_t *A,
		  const pastix_complex64_t *x,
				pastix_complex64_t *y,
				pastix_int_t* rowptr,
				pastix_int_t* colind,
				pastix_int_t* perm){
	
	#pragma omp parallel for
    for( pastix_int_t i=0; i<n; i++)
    {
		pastix_complex64_t dot = 0.0;
		
        for( pastix_int_t j=rowptr[i]-1; j<rowptr[i+1]-1; j++)
        {
            dot += A[j] * x[ perm[colind[j] -1] ];
        }
        
		y[perm[i]] = alpha * dot + beta * y[perm[i]];
    }
}
