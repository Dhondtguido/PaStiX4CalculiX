/**
 *
 * @file z_refine_gmres.c
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
#include "z_refine_functions.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "pastix/order.h"

/**
 *******************************************************************************
 *
 * @ingroup pastix_refine
 *
 * z_gmres_smp - Function computing GMRES iterative refinement.
 *
 *******************************************************************************
 *
 * @param[in] pastix_data
 *          The PaStiX data structure that describes the solver instance.
 *
 * @param[out] x
 *          The solution vector.
 *
 * @param[in] b
 *          The right hand side member (only one).
 *******************************************************************************
 *
 * @return Number of iterations
 *
 *******************************************************************************/
pastix_int_t z_gmres_gpu_smp(pastix_data_t *pastix_data, void *x, void *b)
{
    struct z_solver     solver; 
    Clock               refine_clk;
    pastix_complex64_t *gmHi, *gmH;
    pastix_complex64_t *gmVi, *gmV;
    pastix_complex64_t *gmWi, *gmW, *gmWi_host = NULL;
    pastix_complex64_t *gmcos, *gmsin;
    pastix_complex64_t *gmG;
#if defined(PASTIX_DEBUG_GMRES)
    pastix_complex64_t *dbg_x, *dbg_r, *dbg_G;
#endif
    pastix_complex64_t  tmp;
    pastix_complex64_t*  tmp_ptr;
    pastix_complex64_t *d_b = NULL, *d_x = NULL;
    pastix_fixdbl_t     t0, t3;
    double              eps, resid, resid_b;
    double              norm, normb, normx;
    pastix_int_t        n, im, im1, itermax;
    pastix_int_t        i, j,  ldw, iters;
    int                 outflag, inflag;
    int                 savemem = 0;
    int                 precond = 1;
	spmatrix_t 		   *spm = pastix_data->csc;
    memset( &solver, 0, sizeof(struct z_solver) );
    z_refine_init( &solver, pastix_data );

    /* if ( pastix_data->bcsc->mtxtype == PastixHermitian ) { */
    /*     /\* Check if we need dotu for non hermitian matrices (CEA patch) *\/ */
    /*     solver.dot = &bvev_zdotc_seq; */
    /* } */

    /* Get the parameters */
    n       = pastix_data->bcsc->n;
    im      = pastix_data->iparm[IPARM_GMRES_IM];
    im1     = im + 1;
    itermax = pastix_data->iparm[IPARM_ITERMAX];
    eps     = pastix_data->dparm[DPARM_EPSILON_REFINEMENT];
    ldw     = n;

    if ( !(pastix_data->steps & STEP_NUMFACT) ) {
        precond = 0;
    }

    if ((!precond) || savemem ) {
        ldw = 0;
    }

    gmcos = (pastix_complex64_t *)solver.malloc(im  * sizeof(pastix_complex64_t));
    gmsin = (pastix_complex64_t *)solver.malloc(im  * sizeof(pastix_complex64_t));
    gmG   = (pastix_complex64_t *)solver.malloc(im1 * sizeof(pastix_complex64_t));

    /**
     * H stores the h_{i,j} elements ot the upper hessenberg matrix H (See Alg. 9.5 p 270)
     * V stores the v_{i} vectors
     * W stores the M^{-1} v_{i} vectors to avoid the application of the
     *          preconditioner on the output result (See line 11 of Alg 9.5)
     *
     * If no preconditioner is applied, or the user wants to save memory, W
     * stores only temporarily one vector for the Ax product (ldw is set to 0 to
     * reuse the same vector at each iteration)
     */
     
    gmH = (pastix_complex64_t *)solver.malloc(im * im1 * sizeof(pastix_complex64_t));
    gmV = (pastix_complex64_t *)solver.malloc(n  * im1 * sizeof(pastix_complex64_t));
    if (precond && (!savemem) ) {
        gmW = (pastix_complex64_t *)solver.malloc(n * im  * sizeof(pastix_complex64_t));
    }
    else {
        gmW = (pastix_complex64_t *)solver.malloc(n       * sizeof(pastix_complex64_t));
    }
    if(pastix_data->iparm[IPARM_GPU_NBR] > 0){
#ifdef PASTIX_WITH_CUDA
		gmWi_host = (pastix_complex64_t*) malloc(n * sizeof(pastix_complex64_t));
		cudaMemset( gmH, 0, im * im1 * sizeof(pastix_complex64_t) );
		
		d_x = (pastix_complex64_t *)solver.malloc(n * sizeof(pastix_complex64_t));
		d_b = (pastix_complex64_t *)solver.malloc(n * sizeof(pastix_complex64_t));
		cudaMemcpy( d_x, x, n *  sizeof(pastix_complex64_t), cudaMemcpyHostToDevice);
		cudaMemcpy( d_b, b, n *  sizeof(pastix_complex64_t), cudaMemcpyHostToDevice);
		tmp_ptr = b; b = d_b; d_b = tmp_ptr;
		tmp_ptr = x; x = d_x; d_x = tmp_ptr;
#endif
		}
    else{
		memset( gmH, 0, im * im1 * sizeof(pastix_complex64_t) );
	}

#if defined(PASTIX_DEBUG_GMRES)
    dbg_x = (pastix_complex64_t *)solver.malloc(n   * sizeof(pastix_complex64_t));
    dbg_r = (pastix_complex64_t *)solver.malloc(n   * sizeof(pastix_complex64_t));
    dbg_G = (pastix_complex64_t *)solver.malloc(im1 * sizeof(pastix_complex64_t));
    solver.copy( pastix_data, n, x, dbg_x );
#endif

    normb = solver.norm( pastix_data, n, b );
    normx = solver.norm( pastix_data, n, x );

    clockInit(refine_clk);
    clockStart(refine_clk);

    /**
     * Algorithm from Iterative Methods for Sparse Linear systems, Y. Saad, Second Ed. p267-273
     *
     * The version implemented is the Right preconditioned algorithm.
     */
    outflag = 1;
    iters = 0;

    while (outflag)
    {
        /* Initialize v_{0} and w_{0} */
        gmVi = gmV;
        
        /* Compute r0 = b - A * x */
        solver.copy( pastix_data, n, b, gmVi );
        
        if ( normx > 0. ) {
            //solver.spmv( pastix_data, PastixNoTrans, -1.0, d_x, 1.0, gmVi, streams);
            //solver.spmv( pastix_data, PastixNoTrans, -1., x, 1., gmVi, NULL );
            if(pastix_data->iparm[IPARM_GPU_NBR] > 0){
#ifdef PASTIX_WITH_CUDA
				/*cudaMemcpy(gmWi_host, x, n *  sizeof(pastix_complex64_t), cudaMemcpyDeviceToHost);
				pastix_subtask_applyorder( pastix_data, PastixDouble, PastixDirBackward, n, 1, gmWi_host, n );
				cudaMemcpy(x, gmWi_host, n *  sizeof(pastix_complex64_t), cudaMemcpyHostToDevice);
				
				cudaMemcpy(gmWi_host, gmVi, n *  sizeof(pastix_complex64_t), cudaMemcpyDeviceToHost);
				pastix_subtask_applyorder( pastix_data, PastixDouble, PastixDirBackward, n, 1, gmWi_host, n );
				cudaMemcpy(gmVi, gmWi_host, n *  sizeof(pastix_complex64_t), cudaMemcpyHostToDevice);
				*/
				solver.unblocked_spmv( n, -1., 1., spm->valuesGPU, x, gmVi, spm->colptrGPU, spm->rowptrGPU);
            
				/*cudaMemcpy(gmWi_host, x, n *  sizeof(pastix_complex64_t), cudaMemcpyDeviceToHost);
				pastix_subtask_applyorder( pastix_data, PastixDouble, PastixDirForward, n, 1, gmWi_host, n );
				cudaMemcpy(x, gmWi_host, n *  sizeof(pastix_complex64_t), cudaMemcpyHostToDevice);
				
				cudaMemcpy(gmWi_host, gmVi, n *  sizeof(pastix_complex64_t), cudaMemcpyDeviceToHost);
				pastix_subtask_applyorder( pastix_data, PastixDouble, PastixDirForward, n, 1, gmWi_host, n );
				cudaMemcpy(gmVi, gmWi_host, n *  sizeof(pastix_complex64_t), cudaMemcpyHostToDevice);*/
#endif
			}
			else{
				solver.spmv( pastix_data, PastixNoTrans, -1., x, 1., gmVi );
			}
        }
        
        /* Compute resid = ||r0||_f */
        resid = solver.norm( pastix_data, n, gmVi );
        
        resid_b = resid / normb;
        
        
        /* If residual is small enough, exit */
        if ( resid_b <= eps )
        {
            outflag = 0;
            break;
        }

        /* Compute v0 = r0 / resid */
        tmp = (pastix_complex64_t)( 1.0 / resid );
        solver.scal( pastix_data, n, tmp, gmVi );
        
        if(pastix_data->iparm[IPARM_GPU_NBR] > 0){
#ifdef PASTIX_WITH_CUDA
			cudaMemcpy(gmG, &resid, sizeof(pastix_complex64_t), cudaMemcpyHostToDevice);
#endif
		}
		else{
            gmG[0] = (pastix_complex64_t)resid;
		}
            
        inflag = 1;
        i = -1;
        gmHi = gmH - im1;
        gmWi = gmW - ldw;
        
        while( inflag )
        {
            clockStop( refine_clk );
            t0 = clockGet();
            
            i++;

            /* Set H and W pointers to the beginning of columns i */
            gmHi = gmHi + im1;
            gmWi = gmWi + ldw;
            

           
			solver.copy( pastix_data, n, gmVi, gmWi);
            /* Compute w_{i} = M^{-1} v_{i} */
            if ( precond ) {
				if(pastix_data->iparm[IPARM_GPU_NBR] > 0){
#ifdef PASTIX_WITH_CUDA
					cudaMemcpy(gmWi_host, gmWi, n *  sizeof(pastix_complex64_t), cudaMemcpyDeviceToHost);
					solver.spsv( pastix_data, gmWi_host );
					cudaMemcpy(gmWi, gmWi_host, n *  sizeof(pastix_complex64_t), cudaMemcpyHostToDevice);
#endif
				}
				else
					solver.spsv( pastix_data, gmWi );
            }

            /* v_{i+1} = A (M^{-1} v_{i}) = A w_{i} */
            gmVi += n;
                                    
            //solver.unblocked_spmv( n, 1.0, 0., ValuesGPU, gmWi, gmVi, colptrGPU, rowptrGPU);
            //solver.spmv( pastix_data, PastixNoTrans, 1.0, gmWi, 0., gmVi, streams);
            
            
            if(pastix_data->iparm[IPARM_GPU_NBR] > 0){
#ifdef PASTIX_WITH_CUDA
				/*cudaMemcpy(gmWi_host, gmWi, n *  sizeof(pastix_complex64_t), cudaMemcpyDeviceToHost);
				pastix_subtask_applyorder( pastix_data, PastixDouble, PastixDirBackward, n, 1, gmWi_host, n );
				cudaMemcpy(gmWi, gmWi_host, n *  sizeof(pastix_complex64_t), cudaMemcpyHostToDevice);
				
				cudaMemcpy(gmWi_host, gmVi, n *  sizeof(pastix_complex64_t), cudaMemcpyDeviceToHost);
				pastix_subtask_applyorder( pastix_data, PastixDouble, PastixDirBackward, n, 1, gmWi_host, n );
				cudaMemcpy(gmVi, gmWi_host, n *  sizeof(pastix_complex64_t), cudaMemcpyHostToDevice);
				*/
				solver.unblocked_spmv( n, 1.0, 0., spm->valuesGPU, gmWi, gmVi, spm->colptrGPU, spm->rowptrGPU);
            /*
				cudaMemcpy(gmWi_host, gmWi, n *  sizeof(pastix_complex64_t), cudaMemcpyDeviceToHost);
				pastix_subtask_applyorder( pastix_data, PastixDouble, PastixDirForward, n, 1, gmWi_host, n );
				cudaMemcpy(gmWi, gmWi_host, n *  sizeof(pastix_complex64_t), cudaMemcpyHostToDevice);
				
				cudaMemcpy(gmWi_host, gmVi, n *  sizeof(pastix_complex64_t), cudaMemcpyDeviceToHost);
				pastix_subtask_applyorder( pastix_data, PastixDouble, PastixDirForward, n, 1, gmWi_host, n );
				cudaMemcpy(gmVi, gmWi_host, n *  sizeof(pastix_complex64_t), cudaMemcpyHostToDevice);*/
#endif
			}
			else{
				solver.spmv( pastix_data, PastixNoTrans, 1.0, gmWi, 0., gmVi );
			}
            
            /* Classical Gram-Schmidt */
            for (j=0; j<=i; j++)
            {
                /* Compute h_{j,i} = < v_{i+1}, v_{j} > */
                solver.dot( pastix_data, n, gmVi, gmV + j * n, ((pastix_complex64_t*)gmHi)+j );
                
                /* Compute v_{i+1} = v_{i+1} - h_{j,i} v_{j} */
                if(pastix_data->iparm[IPARM_GPU_NBR] > 0){
#ifdef PASTIX_WITH_CUDA
					cudaMemcpy(&tmp, gmHi + j, sizeof(pastix_complex64_t), cudaMemcpyDeviceToHost);
#endif
				}
				else{
					tmp = gmHi[j];
				}
					
                solver.axpy( pastix_data, n, -tmp ,  gmV + j * n, gmVi );
                
            }
            

            /* Compute || v_{i+1} ||_f */
            norm = solver.norm( pastix_data, n, gmVi );
            
            if(pastix_data->iparm[IPARM_GPU_NBR] > 0){
#ifdef PASTIX_WITH_CUDA
				cudaMemcpy(gmHi+i+1, &norm, sizeof(double), cudaMemcpyHostToDevice);
#endif
			}
			else{
				gmHi[i+1] = norm;
			}
           

            /* Compute v_{i+1} = v_{i+1} / h_{i+1,i} iff h_{i+1,i} is not too small */
            if ( norm > 1e-50 )
            {
                tmp = (pastix_complex64_t)(1.0 / norm);
                solver.scal( pastix_data, n, tmp, gmVi );
            }
            
            
            /* Apply the previous Givens rotation to the new column (should call LAPACKE_zrot_work())*/
			if(pastix_data->iparm[IPARM_GPU_NBR] > 0){
#ifdef PASTIX_WITH_CUDA
				cublasSetPointerMode(*(pastix_data->cublas_handle), CUBLAS_POINTER_MODE_DEVICE);
				
				for (j=0; j<i;j++)
				#if defined(PRECISION_z) ||defined(PRECISION_d)
					cublasZrot(*(pastix_data->cublas_handle), 1, (cuDoubleComplex*) gmHi+j, 1, (cuDoubleComplex*) gmHi+j+1, 1, (double*)(gmcos+j), (cuDoubleComplex*) gmsin+j);
				#else
					cublasZrot(*(pastix_data->cublas_handle), 1, (cuDoubleComplex*) gmHi+j, 1, (cuDoubleComplex*) gmHi+j+1, 1, (float*)(gmcos+j), (cuDoubleComplex*) gmsin+j);
				#endif
				
			#if defined(PRECISION_z) ||defined(PRECISION_d)
				cublasZrotg(*(pastix_data->cublas_handle), (cuDoubleComplex*) gmHi+i, (cuDoubleComplex*) gmHi+i+1, (double*)(gmcos+i), (cuDoubleComplex*) gmsin+i); 
			#else
				cublasZrotg(*(pastix_data->cublas_handle), (cuDoubleComplex*) gmHi+i, (cuDoubleComplex*) gmHi+i+1, (float*)(gmcos+i), (cuDoubleComplex*) gmsin+i); 
			#endif 
				/* Update the residuals (See p. 168, eq 6.35) */
				cudaMemcpy(gmG+i+1, gmG+i, sizeof(pastix_complex64_t), cudaMemcpyDeviceToDevice);
			#if defined(PRECISION_z) ||defined(PRECISION_c)
				cuDoubleComplex negone = make_cuDoubleComplex(-1.0,0);
			#else
				cuDoubleComplex negone = -1.0;
			#endif
				cublasZscal(*(pastix_data->cublas_handle), 1, (cuDoubleComplex*) gmsin+i, (cuDoubleComplex*) gmG+i+1, 1);
				cublasZscal(*(pastix_data->cublas_handle), 1, (cuDoubleComplex*) gmcos+i, (cuDoubleComplex*) gmG+i, 1);
				cublasSetPointerMode(*(pastix_data->cublas_handle), CUBLAS_POINTER_MODE_HOST);
				cublasZscal(*(pastix_data->cublas_handle), 1, (cuDoubleComplex*) (&negone), (cuDoubleComplex*) gmG+i+1, 1);
				cudaMemcpy(&tmp, gmG+i+1, sizeof(pastix_complex64_t), cudaMemcpyDeviceToHost);
#endif
			}
			else{
				for (j=0; j<i;j++)
				{
					tmp = gmHi[j];
					gmHi[j]   = gmcos[j] * tmp       +      gmsin[j]  * gmHi[j+1];
					gmHi[j+1] = gmcos[j] * gmHi[j+1] - conj(gmsin[j]) * tmp;
				}
				
				tmp = csqrt( gmHi[i]   * gmHi[i] +
							 gmHi[i+1] * gmHi[i+1] );

				if ( cabs(tmp) <= eps ) {
					tmp = (pastix_complex64_t)eps;
				}
				gmcos[i] = gmHi[i]   / tmp;
				gmsin[i] = gmHi[i+1] / tmp;
				
				tmp = gmG[i+1] = -gmsin[i] * gmG[i];
				gmG[i]   =  gmcos[i] * gmG[i];
				
				gmHi[i] = gmcos[i] * gmHi[i] + gmsin[i] * gmHi[i+1];
			}
			
            resid = cabs( tmp );


            resid_b = resid / normb;
            iters++;
            
            
            if ( (i+1 >= im) ||
                 (resid_b <= eps) ||
                 (iters >= itermax) )
            {
                inflag = 0;
            }
            
            clockStop((refine_clk));
            t3 = clockGet();
            if ( pastix_data->iparm[IPARM_VERBOSE] > PastixVerboseNot ) {
                solver.output_oneiter( t0, t3, resid_b, iters );
            }
        }

        /* Compute y_m = H_m^{-1} g_m (See p. 169) */
        if(pastix_data->iparm[IPARM_GPU_NBR] > 0){
#ifdef PASTIX_WITH_CUDA
			cublasZtrsv(*(pastix_data->cublas_handle), CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, i+1, (cuDoubleComplex*) gmH, im1, (cuDoubleComplex*) gmG, 1);
#endif
		}
		else{
			cblas_ztrsv( CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, i+1, (cuDoubleComplex*) gmH, im1, (cuDoubleComplex*) gmG, 1 );
		}

        /**
         * Compute x_m = x_0 + M^{-1} V_m y_m
         *             = x_0 +        W_m y_m
         */
        if (precond && savemem) {
            /**
             * Since we saved memory, we do not have (M^{-1} V_m) stored,
             * thus we compute:
             *     w = V_m y_m
             *     w = M^{-1} (V_m y_m)
             *     x = x0 + (M^{-1} (V_m y_m))
             */
            solver.gemv( pastix_data, n, i+1, 1.0, gmV, n, (pastix_complex64_t*)gmG, 0., gmW );
            solver.spsv( pastix_data, gmW );
            solver.axpy( pastix_data, n, 1.,  gmW, x );
        }
        else {
            /**
             * Since we did not saved memory, we do have (M^{-1} V_m) stored in
             * W_m if precond is true, thus we compute:
             *     x = x0 + W_m y_m, if precond
             *     x = x0 + V_m y_m, if not precond
             */
             
            gmWi = precond ? gmW : gmV;
            
            solver.gemv( pastix_data, n, i+1, 1.0, gmWi, n, (pastix_complex64_t*)gmG, 1.0, x );
            
        }

        if ((resid_b <= eps) || (iters >= itermax))
        {
            outflag = 0;
        }
    }
    

    clockStop( refine_clk );
    t3 = clockGet();

    if(pastix_data->iparm[IPARM_GPU_NBR] > 0){
#ifdef PASTIX_WITH_CUDA
		tmp_ptr = b; b = d_b; d_b = tmp_ptr;
		tmp_ptr = x; x = d_x; d_x = tmp_ptr;
		cudaMemcpy(x, d_x, n * sizeof(pastix_complex64_t), cudaMemcpyDeviceToHost);
		free(gmWi_host);
		solver.free(d_b);
		solver.free(d_x);
#endif
	}

    solver.output_final( pastix_data, resid_b, iters, t3, x, x );


    solver.free(gmcos);
    solver.free(gmsin);
    solver.free(gmG);
    solver.free(gmH);
    solver.free(gmV);
    solver.free(gmW);
#if defined(PASTIX_DEBUG_GMRES)
    solver.free(dbg_x);
    solver.free(dbg_r);
    solver.free(dbg_G);
#endif

	if(iters >= itermax)
		return -1;

    return iters;
}
