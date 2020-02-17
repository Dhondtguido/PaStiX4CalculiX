/**
 *
 * @file pastix_task_refine.c
 *
 * PaStiX refinement functions implementations.
 *
 * @copyright 2015-2018 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @version 6.0.1
 * @author Mathieu Faverge
 * @author Pierre Ramet
 * @author Xavier Lacoste
 * @author Gregoire Pichon
 * @author Theophile Terraz
 * @date 2018-07-16
 *
 **/
#include "common.h"
#include "bcsc.h"
#include "z_refine_functions.h"
#include "c_refine_functions.h"
#include "d_refine_functions.h"
#include "s_refine_functions.h"
#include "pastix/order.h"
#include <blend/solver.h>
#include <spm.h>
#include <bcsc/bcsc.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "kernels/gpus/LightSpMV-1.0/src/cLightSpMV.h"

#include <parsec.h>
#include <parsec/data.h>
#include <parsec/data_distribution.h>
#if defined(PASTIX_WITH_CUDA)
#include <parsec/devices/cuda/dev_cuda.h>
#endif
#include "parsec/utils/zone_malloc.h"

extern gpu_device_t* gpu_device;
extern char* gpu_base;

/**
 *******************************************************************************
 *
 * @ingroup pastix_dev_refine
 *
 * @brief Select the refinement function to call depending on the matrix type
 * and the precision
 *
 *******************************************************************************/
static pastix_int_t (*sopalinRefine[5][4])(pastix_data_t *pastix_data, void *x, void *b) =
{
    //  PastixRefineGMRES_GPU
    {
        s_gmres_gpu_smp,
        d_gmres_gpu_smp,
        c_gmres_gpu_smp,
        z_gmres_gpu_smp
    },
    //  PastixRefineGMRES
    {
        s_gmres_smp,
        d_gmres_smp,
        c_gmres_smp,
        z_gmres_smp
    },
    //  PastixRefineCG
    {
        s_grad_smp,
        d_grad_smp,
        c_grad_smp,
        z_grad_smp
    },
    //  PastixRefineSR
    {
        s_pivot_smp,
        d_pivot_smp,
        c_pivot_smp,
        z_pivot_smp
    },
    //  PastixRefineBiCGSTAB
    {
        s_bicgstab_smp,
        d_bicgstab_smp,
        c_bicgstab_smp,
        z_bicgstab_smp
    }
};

/**
 *******************************************************************************
 *
 * @ingroup pastix_refine
 *
 * @brief Perform the iterative refinement without apply the permutations.
 *
 * This routine is affected by the following parameters:
 *   IPARM_REFINEMENT, DPARM_EPSILON_REFINEMENT
 *
 *******************************************************************************
 *
 * @param[in] pastix_data
 *          The PaStiX data structure that describes the solver instance.
 *
 * @param[in] n
 *          The size of system to solve, and the number of rows of both
 *          matrices b and x.
 *
 * @param[in] nrhs
 *          The number of right hand side members, and the number of columns of
 *          b and x.
 *
 * @param[inout] b
 *          The right hand side matrix of size ldb-by-nrhs.
 *          B is noted as inout, as permutation might be performed on the
 *          matrix. On exit, the matrix is restored as it was on entry.
 *
 * @param[in] ldb
 *          The leading dimension of the matrix b. ldb >= n.
 *
 * @param[inout] x
 *          The matrix x of size ldx-by-nrhs.
 *          On entry, the initial guess x0 for the refinement step, that may be
 *          the solution returned by the solve step or any other initial guess.
 *          On exit, contains the final solution after the iterative refinement.
 *
 * @param[in] ldx
 *          The leading dimension of the matrix x. ldx >= n.
 *
 *******************************************************************************
 *
 * @retval PASTIX_SUCCESS on successful exit,
 * @retval PASTIX_ERR_BADPARAMETER if one parameter is incorrect,
 *
 *******************************************************************************/
int
pastix_subtask_refine( pastix_data_t *pastix_data,
                       pastix_int_t n, pastix_int_t nrhs,
                             void *b, pastix_int_t ldb,
                             void *x, pastix_int_t ldx)
{
    pastix_int_t   *iparm = pastix_data->iparm;
    pastix_bcsc_t  *bcsc  = pastix_data->bcsc;

    if (nrhs > 1)
    {
        errorPrintW("Refinement works only with 1 rhs, We will iterate on each RHS one by one\n");
    }

    if ( (pastix_data->schur_n > 0) && (iparm[IPARM_SCHUR_SOLV_MODE] != PastixSolvModeLocal))
    {
        fprintf(stderr, "Refinement is not available with Schur complement when non local solve is required\n");
        return PASTIX_ERR_BADPARAMETER;
    }

    /* Prepare the refinement threshold, if not set by the user */
    if ( pastix_data->dparm[DPARM_EPSILON_REFINEMENT] < 0. ) {
        if ( (bcsc->flttype == PastixFloat) ||
             (bcsc->flttype == PastixComplex32) ) {
            pastix_data->dparm[DPARM_EPSILON_REFINEMENT] = 1e-12;
        }
        else {
            pastix_data->dparm[DPARM_EPSILON_REFINEMENT] = 1e-12;
        }
    }
    
    void *xptr = (char *)(x);
	void *bptr = (char *)(b);

	pastix_int_t (*refinefct)(pastix_data_t *, void *, void *) = sopalinRefine[iparm[IPARM_REFINEMENT]][1];
	
	size_t shiftx, shiftb;
	int i;

	shiftx = ldx * pastix_size_of( PastixDouble );
	shiftb = ldb * pastix_size_of( PastixDouble );

	for(i=0; i<nrhs; i++, xptr += shiftx, bptr += shiftb ) {
		pastix_int_t it;
		it = refinefct( pastix_data, xptr, bptr);
		if(it == -1)
			return -1;
		pastix_data->iparm[IPARM_NBITER] = pastix_imax( it, pastix_data->iparm[IPARM_NBITER] );
	}
	
    (void)n;
    return PASTIX_SUCCESS;
}

/**
 *******************************************************************************
 *
 * @ingroup pastix_users
 *
 * @brief Perform iterative refinement.
 *
 * This routine performs the permutation of x, and b before and after the
 * iterative refinement solution. To prevent extra permuation to happen, see
 * pastix_subtask_refine().
 * This routine is affected by the following parameters:
 *   IPARM_REFINEMENT, DPARM_EPSILON_REFINEMENT
 *
 *******************************************************************************
 *
 * @param[in] pastix_data
 *          The PaStiX data structure that describes the solver instance.
 *
 * @param[in] n
 *          The size of system to solve, and the number of rows of both
 *          matrices b and x.
 *
 * @param[in] nrhs
 *          The number of right hand side members, and the number of columns of
 *          b and x.
 *
 * @param[inout] b
 *          The right hand side matrix of size ldb-by-nrhs.
 *          B is noted as inout, as permutation might be performed on the
 *          matrix. On exit, the matrix is restored as it was on entry.
 *
 * @param[in] ldb
 *          The leading dimension of the matrix b. ldb >= n.
 *
 * @param[inout] x
 *          The matrix x of size ldx-by-nrhs.
 *          On entry, the initial guess x0 for the refinement step, that may be
 *          the solution returned by the solve step or any other initial guess.
 *          On exit, contains the final solution after the iterative refinement.
 *
 * @param[in] ldx
 *          The leading dimension of the matrix x. ldx >= n.
 *
 *******************************************************************************
 *
 * @retval PASTIX_SUCCESS on successful exit,
 * @retval PASTIX_ERR_BADPARAMETER if one parameter is incorrect,
 *
 *******************************************************************************/
int
pastix_task_refine( pastix_data_t *pastix_data,
                    pastix_int_t n, pastix_int_t nrhs,
                    void *b, pastix_int_t ldb,
                    void *x, pastix_int_t ldx)
{
	spmatrix_t* spm = pastix_data->csc;
    pastix_int_t  *iparm = pastix_data->iparm;
    pastix_bcsc_t *bcsc  = pastix_data->bcsc;
    char failed = 0;
    int rc;
    void* tmpBcscValues = NULL;
    double timer;
    char GPUtemp = iparm[IPARM_GPU_NBR];
    if( pastix_data->iparm[IPARM_FLOAT] == 3)
		iparm[IPARM_GPU_NBR] = 0;
    
    if ( (pastix_data->schur_n > 0) && (iparm[IPARM_SCHUR_SOLV_MODE] != PastixSolvModeLocal))
    {
        fprintf(stderr, "Refinement is not available with Schur complement when non local solve is required\n");
        return PASTIX_ERR_BADPARAMETER;
    }

    /* Prepare the refinement threshold, if not set by the user */
    if ( pastix_data->dparm[DPARM_EPSILON_REFINEMENT] < 0. ) {
        if ( (bcsc->flttype == PastixFloat) ||
             (bcsc->flttype == PastixComplex32) ) {
            pastix_data->dparm[DPARM_EPSILON_REFINEMENT] = 1e-12;
        }
        else {
            pastix_data->dparm[DPARM_EPSILON_REFINEMENT] = 1e-12;
        }
    }
    
    clockStart(timer);
    {
    

		if( spm->mtxtype == SpmGeneral ){
			tmpBcscValues = bcsc->Uvalues;
			bcsc->Uvalues = spm->values;
		}
		else{
			tmpBcscValues = bcsc->Lvalues;
			bcsc->Lvalues = bcsc->Uvalues = spm->values;
		}
		

		/* Compute P * b */
		rc = pastix_subtask_applyorder( pastix_data, PastixDouble,
										PastixDirForward, bcsc->gN, nrhs, b, ldb );
		if( rc != PASTIX_SUCCESS ) {
			return rc;
		}

		/* Compute P * x */
		rc = pastix_subtask_applyorder( pastix_data, PastixDouble,
										PastixDirForward, bcsc->gN, nrhs, x, ldx );
		if( rc != PASTIX_SUCCESS ) {
			return rc;
		}
		/* Performe the iterative refinement */
		rc = pastix_subtask_refine( pastix_data, n, nrhs, b, ldb, x, ldx );
		if( rc == -1)
			failed = 1;
		else if( rc != PASTIX_SUCCESS ) {
			return rc;
		}
		/* Compute P * b */
		rc = pastix_subtask_applyorder( pastix_data, PastixDouble,
										PastixDirBackward, bcsc->gN, nrhs, b, ldb );
		if( rc != PASTIX_SUCCESS ) {
			return rc;
		}

		/* Compute P * x */
		rc = pastix_subtask_applyorder( pastix_data, PastixDouble,
										PastixDirBackward, bcsc->gN, nrhs, x, ldx );
		if( rc != PASTIX_SUCCESS ) {
			return rc;
		}
		if(pastix_data->iparm[IPARM_GPU_NBR] > 0){
	#ifdef PASTIX_WITH_CUDA
			destroyLightSpMV();
			
			gpu_base = gpu_device->memory->base;

/*
			zone_free( gpu_device->memory, spm->valuesGPU );
			zone_free( gpu_device->memory, spm->colptrGPU );
			zone_free( gpu_device->memory, spm->rowptrGPU );*/
			/*
			cudaFree(spm->valuesGPU);
			cudaFree(spm->colptrGPU);
			cudaFree(spm->rowptrGPU);
			*/
			spm->valuesGPU = NULL;
			spm->colptrGPU = NULL;
			spm->rowptrGPU = NULL;
	#endif
		}
		
		if( spm->mtxtype == SpmGeneral ){
			bcsc->Uvalues = tmpBcscValues;
		}
		else{
			bcsc->Lvalues = bcsc->Uvalues = tmpBcscValues;
		}
		
		if( pastix_data->iparm[IPARM_FLOAT] == 3)
			iparm[IPARM_GPU_NBR] = GPUtemp;

	}
    clockStop(timer);

    pastix_data->dparm[DPARM_REFINE_TIME] = clockVal(timer);
    if (iparm[IPARM_VERBOSE] > PastixVerboseNot) {
        pastix_print( 0, 0, OUT_TIME_REFINE,
                      pastix_data->dparm[DPARM_REFINE_TIME] );
    }

    (void)n;
    
    if(failed)
		return -1;
    return PASTIX_SUCCESS;
}
