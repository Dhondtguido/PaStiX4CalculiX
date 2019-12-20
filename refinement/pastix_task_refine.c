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

/**
 *******************************************************************************
 *
 * @ingroup pastix_dev_refine
 *
 * @brief Select the refinement function to call depending on the matrix type
 * and the precision
 *
 *******************************************************************************/
static pastix_int_t (*sopalinRefine[5][4])(pastix_data_t *pastix_data, void *x, void *b, spmatrix_t *spm) =
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
                             void **b, pastix_int_t ldb,
                             void **x, pastix_int_t ldx,
					   spmatrix_t* spm )
{
    pastix_int_t   *iparm = pastix_data->iparm;
    pastix_bcsc_t  *bcsc  = pastix_data->bcsc;
    double timer;

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
    
    void *xptr = (char *)(*x);
	void *bptr = (char *)(*b);

    clockStart(timer);
    {
        pastix_int_t (*refinefct)(pastix_data_t *, void *, void *, spmatrix_t *) = sopalinRefine[iparm[IPARM_REFINEMENT]][1];
        
        size_t shiftx, shiftb;
        int i;

        shiftx = ldx * pastix_size_of( PastixDouble );
        shiftb = ldb * pastix_size_of( PastixDouble );

        for(i=0; i<nrhs; i++, xptr += shiftx, bptr += shiftb ) {
            pastix_int_t it;
            it = refinefct( pastix_data, xptr, bptr, spm);
            pastix_data->iparm[IPARM_NBITER] = pastix_imax( it, pastix_data->iparm[IPARM_NBITER] );
        }
	}
    clockStop(timer);

    pastix_data->dparm[DPARM_REFINE_TIME] = clockVal(timer);
    if (iparm[IPARM_VERBOSE] > PastixVerboseNot) {
        pastix_print( 0, 0, OUT_TIME_REFINE,
                      pastix_data->dparm[DPARM_REFINE_TIME] );
    }
    
    if ( iparm[IPARM_GPU_NBR] > 0 ) {
#ifdef PASTIX_WITH_CUDA
		cudaFree(spm->valuesDouble);
		spm->valuesDouble = NULL;
		cudaFree(spm->colptr);
		spm->colptr = NULL;
		cudaFree(spm->rowptr);
		spm->rowptr = NULL;
		cudaFree(pastix_data->ordemesh->permtab);
		pastix_data->ordemesh->permtab = NULL;
#endif
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
                    void **b, pastix_int_t ldb,
                    void **x, pastix_int_t ldx,
                    spmatrix_t* spm )
{
    pastix_int_t  *iparm = pastix_data->iparm;
    pastix_bcsc_t *bcsc  = pastix_data->bcsc;
    int rc;
    double timer;

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
    
    pastix_data->iparm[IPARM_GPU_NBR] = 0;

    clockStart(timer);
    {
		
       if(iparm[66] == 2){
			/*
			double *xptrD = (double*) malloc(sizeof(double) * n);
		//	double *bptrD = (double*) malloc(sizeof(double) * n);
			
			#pragma omp parallel for
			for(int i = 0; i < n; i++){
				xptrD[i] = (double) (((float*) *x)[i]);
		//		bptrD[i] = (double) (((float*) *b)[i]);
			}
			
			free(*x);
		//	free(*b);
			
			*x = (char*) xptrD;
		//	*b = (char*) bptrD;
			*/
			
			if ( iparm[IPARM_GPU_NBR] <= 0 ) {
				int numElements = bcsc->numElements;
				
				double* L_new = (double*) malloc(sizeof(double) * numElements);
				float* L_old = (float*) bcsc->Lvalues;			
				bcsc->Lvalues = L_new;
				#pragma omp simd
				for(int i = 0; i < numElements; i++){
					L_new[i] = (double) L_old[i];
				}
				free(L_old);
				
				float* U_old = (float*) bcsc->Uvalues;
				
				if(bcsc->mtxtype != SpmGeneral){
					bcsc->Uvalues = bcsc->Lvalues;
				}
				else if(bcsc->Uvalues){
					double* U_new = (double*) malloc(sizeof(double) * numElements);
					bcsc->Uvalues = U_new;
					#pragma omp simd
					for(int i = 0; i < numElements; i++){
						U_new[i] = (double) U_old[i];
					}
					free(U_old);
				}
			}
			else{
#ifdef PASTIX_WITH_CUDA
				
				createLightSpMV(n, spm->gnnz);
				
				double* permValuesT = (double*) malloc(spm->gnnz * sizeof(double));
				double* dValues = (double*) spm->valuesDouble;
				
				pastix_int_t* temp = (pastix_int_t*) calloc(spm->n, sizeof(pastix_int_t));
				
				for(int i = 0; i < spm->n; i++){
					for(int j = spm->colptr[i] - 1; j < spm->colptr[i+1] - 1 ; j++){
						pastix_int_t target = spm->colptr[spm->rowptr[j]-1]-1 + (temp[spm->rowptr[j]-1]++);
						permValuesT[target] = dValues[j];
					}
				}
				
				free(temp);
				
				pastix_int_t* perm = pastix_data->ordemesh->permtab;
				pastix_int_t* newColptr = (pastix_int_t*) malloc((spm->n+1) * sizeof(pastix_int_t));
				pastix_int_t* newRowptr = (pastix_int_t*) malloc((spm->gnnz) * sizeof(pastix_int_t));
				
				newColptr[0] = 1;
				for(pastix_int_t i = 0; i < spm->n; i++){
					newColptr[i+1] = newColptr[i] + spm->colptr[pastix_data->ordemesh->peritab[i]+1] - spm->colptr[pastix_data->ordemesh->peritab[i]];
				}
				
				for(pastix_int_t i = 0; i < spm->nnz; i++){
					newRowptr[i] = perm[spm->rowptr[i]-1] + 1;
					dValues[i] = 0.0;
				}
				
				for(int i = 0; i < spm->n; i++){
					for(pastix_int_t j = spm->colptr[i] - 1; j < spm->colptr[i+1] - 1 ; j++){
						pastix_int_t target = newColptr[pastix_data->ordemesh->permtab[i]] - 1 + j - spm->colptr[i] + 1;
						
						spm->rowptr[target] = newRowptr[j];
						dValues[target] = permValuesT[j];
					}
				}
				
				/*
				
				bcsc_cblk_t* firstCblk = bcsc->cscftab;
				float* L_old = (float*) bcsc->Lvalues;
				for(pastix_int_t i = 0; i < 10; i++){
					if(firstCblk->coltab[i] < firstCblk->coltab[i+1]){
						printf("bcsc:\n");
						printf("col = %ld    row = %ld\n", i+1, bcsc->rowtab[firstCblk->coltab[i]]+1);
						printf("%.15f\n", L_old[firstCblk->coltab[i]]);
						
						pastix_int_t minimum = 1000000000;
						pastix_int_t index = 0;
						for(pastix_int_t j = newColptr[i]; j < newColptr[i+1]; j++){
							if(spm->rowptr[j] < minimum){
								minimum = spm->rowptr[j];
								index = j;
							}
						}	
						printf("big csc:\n");
						printf("col = %ld    row = %ld\n", i+1, minimum);
						printf("%.15lf\n", dValues[index]);
						printf("\n");
					}
					
					printf("\n");
					

				}*/
				
				free(spm->colptr);
				free(newRowptr);
				spm->colptr = newColptr;
				//spm->rowptr = newRowptr;
				free(permValuesT);
				
				void* ValuesGPU;
				cudaMalloc((void**) &ValuesGPU, spm->nnzexp * sizeof(double));
				cudaMemcpy(ValuesGPU, spm->valuesDouble, spm->nnzexp * sizeof(double), cudaMemcpyHostToDevice);
				free(spm->valuesDouble);
				spm->valuesDouble = ValuesGPU;
				
				pastix_int_t* colptrGPU;
				cudaMalloc((void**) &colptrGPU, (spm->n+1) * sizeof(pastix_int_t));
				cudaMemcpy(colptrGPU, spm->colptr, (spm->n+1) * sizeof(pastix_int_t), cudaMemcpyHostToDevice);
				free(spm->colptr);
				spm->colptr = colptrGPU;
				
				pastix_int_t* rowptrGPU;
				cudaMalloc((void**) &rowptrGPU, spm->nnzexp * sizeof(pastix_int_t));
				cudaMemcpy(rowptrGPU, spm->rowptr, spm->nnzexp * sizeof(pastix_int_t), cudaMemcpyHostToDevice);
				free(spm->rowptr);
				spm->rowptr = rowptrGPU;
				
				pastix_int_t* permGPU;
				cudaMalloc((void**) &permGPU, spm->n * sizeof(pastix_int_t));
				cudaMemcpy(permGPU, pastix_data->ordemesh->permtab, spm->n * sizeof(pastix_int_t), cudaMemcpyHostToDevice);
				free(pastix_data->ordemesh->permtab);
				pastix_data->ordemesh->permtab = permGPU;
#endif
			}
			/*void* ValuesGPU;
			pastix_int_t* rowtabGPU;
			
			cudaMalloc((void**) &ValuesGPU, pastix_data->bcsc->numElements * sizeof(pastix_complex64_t));
			if(pastix_data->bcsc->mtxtype == PastixGeneral && pastix_data->bcsc->Uvalues != NULL)
			{
				cudaMemcpy(ValuesGPU, pastix_data->bcsc->Uvalues, pastix_data->bcsc->numElements * sizeof(pastix_complex64_t), cudaMemcpyHostToDevice);
				pastix_data->bcsc->Uvalues = ValuesGPU;
			}
			else
			{
				cudaMemcpy(ValuesGPU, pastix_data->bcsc->Lvalues, pastix_data->bcsc->numElements * sizeof(pastix_complex64_t), cudaMemcpyHostToDevice);
				pastix_data->bcsc->Lvalues = ValuesGPU;
			}
			
			cudaMalloc((void**) &rowtabGPU, pastix_data->bcsc->numElements * sizeof(pastix_int_t));
			cudaMemcpy(rowtabGPU, pastix_data->bcsc->rowtab, pastix_data->bcsc->numElements * sizeof(pastix_int_t), cudaMemcpyHostToDevice);
			pastix_data->bcsc->rowtab = rowtabGPU;
			
			for(int i = 0; i < pastix_data->bcsc->cscfnbr; i++){
				bcsc_cblk_t* cblk = (pastix_data->bcsc->cscftab)+i;
				pastix_int_t* coltabGPU;
				cudaMalloc((void**) &coltabGPU, (cblk->colnbr + 1) * sizeof(pastix_int_t));
				cudaMemcpy(coltabGPU, cblk->coltab, (cblk->colnbr + 1) * sizeof(pastix_int_t), cudaMemcpyHostToDevice);
				cblk->coltab = coltabGPU;
			}*/
			/*
			
			SolverCblk* cblktab = pastix_data->solvmatr->cblktab;
			int numCblks = pastix_data->solvmatr->cblknbr;
			
			for(int i = 0; i < numCblks; i++){
				int cblksize = (cblktab[i].lcolnum - cblktab[i].fcolnum + 1) * cblktab[i].stride;

				float* L_old = (float*) (cblktab[i].lcoeftab);
				double* L_new = (double*) malloc(sizeof(double) * cblksize);
				#pragma omp simd
				for( int j = 0; j < cblksize; j++){
					L_new[j] = (double) L_old[j];
				}
				free(L_old);
				cblktab[i].lcoeftab = L_new;
				
				float* U_old = (float*) (cblktab[i].ucoeftab);
				if(U_old){
					double* U_new = (double*) malloc(sizeof(double) * cblksize);
					#pragma omp simd
					for( int j = 0; j < cblksize; j++){
						U_new[j] = (double) U_old[j];
					}
					free(U_old);
					cblktab[i].ucoeftab = U_new;
				}
			}*/
		}
	}
	
	bcsc->flttype = PastixDouble;
    clockStop(timer);
    
    
    
    if (iparm[IPARM_VERBOSE] > PastixVerboseNot) {
        pastix_print( 0, 0, OUT_TIME_CAST,
                      clockVal(timer) );
    }



    /* Compute P * b */
    rc = pastix_subtask_applyorder( pastix_data, bcsc->flttype,
                                    PastixDirForward, bcsc->gN, nrhs, *b, ldb );
    if( rc != PASTIX_SUCCESS ) {
        return rc;
    }

    /* Compute P * x */
    rc = pastix_subtask_applyorder( pastix_data, bcsc->flttype,
                                    PastixDirForward, bcsc->gN, nrhs, *x, ldx );
    if( rc != PASTIX_SUCCESS ) {
        return rc;
    }
	if(iparm[66] == 2)
		bcsc->flttype = PastixFloat;
    /* Performe the iterative refinement */
    rc = pastix_subtask_refine( pastix_data, n, nrhs, b, ldb, x, ldx, spm );
    if( rc != PASTIX_SUCCESS ) {
        return rc;
    }
	if(iparm[66] == 2)
		bcsc->flttype = PastixDouble;
    /* Compute P * b */
    rc = pastix_subtask_applyorder( pastix_data, bcsc->flttype,
                                    PastixDirBackward, bcsc->gN, nrhs, *b, ldb );
    if( rc != PASTIX_SUCCESS ) {
        return rc;
    }

    /* Compute P * x */
    rc = pastix_subtask_applyorder( pastix_data, bcsc->flttype,
                                    PastixDirBackward, bcsc->gN, nrhs, *x, ldx );
    if( rc != PASTIX_SUCCESS ) {
        return rc;
    }

    pastix_data->iparm[IPARM_GPU_NBR] = 1;
    (void)n;
    return PASTIX_SUCCESS;
}
