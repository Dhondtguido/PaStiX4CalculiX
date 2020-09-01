/**
 *
 * @file z_refine_functions.c
 *
 * PaStiX refinement functions implementations.
 *
 * @copyright 2015-2018 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @version 6.0.1
 * @author Mathieu Faverge
 * @author Pierre Ramet
 * @author Theophile Terraz
 * @author Xavier Lacoste
 * @date 2018-07-16
 * @precisions normal z -> c d s
 *
 **/
#include "common.h"
#include "cblas.h"
#include "bcsc.h"
#include "bvec.h"
#include "bcsc_z.h"
#include "sopalin_data.h"
#include "z_refine_functions.h"
#include "cpu_z_spmv.h"
#include "gpu_z_spmv.h"

/**
 *******************************************************************************
 *
 * @ingroup pastix_dev_refine
 *
 * @brief Print statistics about one iteration
 *
 *******************************************************************************
 *
 * @param[in] t0
 *          The clock value at the beginning of the iteration
 *
 * @param[in] tf
 *          The clock value at the end of the iteration
 *
 * @param[in] err
 *          The backward error after the iteration
 *
 * @param[in] nb_iters
 *          Current number of refinement iterations
 *
 *******************************************************************************/
void z_refine_output_oneiter( double t0, double tf, double err, pastix_int_t nb_iters )
{
    double stt;

    stt = tf - t0;
    fprintf(stdout, OUT_ITERREFINE_ITER, (int)nb_iters);
    fprintf(stdout, OUT_ITERREFINE_TTT, stt);
    fprintf(stdout, OUT_ITERREFINE_ERR, err);
}

/**
 *******************************************************************************
 *
 * @ingroup pastix_dev_refine
 *
 * @brief Final output
 *
 *******************************************************************************
 *
 * @param[in] pastix_data
 *          The PaStiX data structure that describes the solver instance.
 *
 * @param[in] err
 *          The final backward error
 *
 * @param[in] nb_iters
 *          The final number of iterations
 *
 * @param[in] tf
 *          The final clock value
 *
 * @param[inout] x
 *          The vector that is to be overwritten by gmresx
 *
 * @param[in] gmresx
 *          The final solution
 *
 *******************************************************************************/
void z_refine_output_final( pastix_data_t      *pastix_data,
                            pastix_complex64_t  err,
                            pastix_int_t        nb_iters,
                            double              tf,
                            void               *x,
                            pastix_complex64_t *gmresx )
{
    (void)pastix_data;
    (void)err;
    (void)nb_iters;
    (void)tf;
    (void)x;
    (void)gmresx;
    pastix_data->dparm[DPARM_RELATIVE_ERROR] = err;
}

/**
 *******************************************************************************
 *
 * @ingroup pastix_dev_refine
 *
 * @brief Initiate functions pointers to define basic operations
 *
 *******************************************************************************
 *
 * @param[out] solver
 *          The structure to be filled
 *
 *******************************************************************************/
void z_refine_init( struct z_solver *solver, pastix_data_t *pastix_data )
{
    pastix_scheduler_t sched = pastix_data->iparm[IPARM_SCHEDULER];
    int num_gpus = pastix_data->iparm[IPARM_GPU_NBR];
    
    /* Allocations */
    if ( num_gpus > 0){
		solver->malloc  = &bvec_malloc_cuda;
		solver->free    = &bvec_free_cuda;
	}
	else{
		solver->malloc  = &bvec_malloc;
		solver->free    = &bvec_free;
	}

    /* Output */
    solver->output_oneiter = &z_refine_output_oneiter;
    solver->output_final   = &z_refine_output_final;

    /* Basic operations */
#if defined(PRECISION_d)
    if(pastix_data->iparm[66] == 2)
		solver->spsv = &bcsc_DSspsv;
	else
		solver->spsv = &bcsc_zspsv;
#else
	solver->spsv = &bcsc_zspsv;
#endif

	if ( num_gpus > 0){
#ifdef PASTIX_WITH_CUDA
		solver->spmv = &bcsc_zspmv;
		solver->unblocked_spmv = &gpu_z_spmv;
        solver->copy = &bvec_zcopy_cuda;
        solver->dot  = &bvec_zdotc_cuda;
        solver->axpy = &bvec_zaxpy_cuda;
        solver->scal = &bvec_zscal_cuda;
        solver->norm = &bvec_znrm2_cuda;
        solver->gemv = &bvec_zgemv_cuda;
#endif
	}
    else if ( sched == PastixSchedSequential ) {
        solver->spmv = &bcsc_zspmv;
		solver->unblocked_spmv = &cpu_z_spmv;
        solver->copy = &bvec_zcopy_seq;
        solver->dot  = &bvec_zdotc_seq;
        solver->axpy = &bvec_zaxpy_seq;
        solver->scal = &bvec_zscal_seq;
        solver->norm = &bvec_znrm2_seq;
        solver->gemv = &bvec_zgemv_seq;
    } else {
        solver->spmv = &bcsc_zspmv;
		solver->unblocked_spmv = &cpu_z_spmv;
        solver->copy = &bvec_zcopy_smp;
        solver->dot  = &bvec_zdotc_smp;
        solver->axpy = &bvec_zaxpy_smp;
        solver->scal = &bvec_zscal_smp;
        solver->norm = &bvec_znrm2_smp;
        solver->gemv = &bvec_zgemv_smp;
    }
}
