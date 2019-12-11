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
pastix_int_t z_gmres_gpu_smp(pastix_data_t *pastix_data, void *x, void *b, spmatrix_t *spm)
{
    struct z_solver     solver; 
    Clock               refine_clk;
    cuDoubleComplex *gmHi, *gmH;
    pastix_complex64_t *gmVi, *gmV;
    pastix_complex64_t *gmVi_buffer;
    pastix_complex64_t *gmWi, *gmW;
    pastix_complex64_t *gmWi_host, *gmW_host;
    cuDoubleComplex *gmsin;
    cuDoubleComplex *gmcos;
    cuDoubleComplex *gmG;
    pastix_complex64_t buffer;
    cuDoubleComplex *bufferG;
#if defined(PASTIX_DEBUG_GMRES)
    pastix_complex64_t *dbg_x, *dbg_r, *dbg_G;
#endif
    pastix_complex64_t  tmp;
    pastix_fixdbl_t     t0, t3;
    double              eps, resid, resid_b;
    double              norm, normb, normx;
    pastix_int_t        n, im, im1, itermax;
    pastix_int_t        i, j,  ldw, iters;
    int                 outflag, inflag;
    int                 savemem = 0;
    int                 precond = 1;

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
    
    cudaStream_t streams[64];
    for(int x = 0; x < 64; x++){
			cudaStreamCreate(streams + x);
	}
        

	void* d_x;
	void* d_b;
	
	cudaMalloc((void**)&d_x, n * sizeof(pastix_complex64_t));
	cudaMalloc((void**)&d_b, n * sizeof(pastix_complex64_t));
	
	cudaMemcpy(d_x, x, n * sizeof(pastix_complex64_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, n * sizeof(pastix_complex64_t), cudaMemcpyHostToDevice);

    if ( !(pastix_data->steps & STEP_NUMFACT) ) {
        precond = 0;
    }

    if ((!precond) || savemem ) {
        ldw = 0;
    }

	cudaMalloc((void**)&gmcos, im  * sizeof(pastix_complex64_t));
	cudaMalloc((void**)&gmsin, im  * sizeof(pastix_complex64_t));
	cudaMalloc((void**)&gmG, im1  * sizeof(pastix_complex64_t));
	cudaMalloc((void**)&bufferG, 2  * sizeof(pastix_complex64_t));

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
	cudaMalloc((void**)&gmH, im * im1 * sizeof(pastix_complex64_t));
	cudaMalloc((void**)&gmV, n  * im1 * sizeof(pastix_complex64_t));
	
    if (precond && (!savemem) ) {
		cudaMalloc((void**)&gmW, n  * im * sizeof(pastix_complex64_t));
    }
    else {
		cudaMalloc((void**)&gmW, n * sizeof(pastix_complex64_t));
    }
    gmW_host = (pastix_complex64_t*) malloc(n  * im * sizeof(pastix_complex64_t));
    cudaMemset( gmH, 0, im * im1 * sizeof(pastix_complex64_t) );
    
    
    gmVi_buffer = (pastix_complex64_t*) malloc(n * sizeof(pastix_complex64_t));
/*
#if defined(PASTIX_DEBUG_GMRES)
    dbg_x = (pastix_complex64_t *)solver.malloc(n   * sizeof(pastix_complex64_t));
    dbg_r = (pastix_complex64_t *)solver.malloc(n   * sizeof(pastix_complex64_t));
    dbg_G = (pastix_complex64_t *)solver.malloc(im1 * sizeof(pastix_complex64_t));
    solver.copy( pastix_data, n, x, dbg_x );
#endif
*/
	void* ValuesGPU;
	cudaMalloc((void**) &ValuesGPU, spm->nnzexp * sizeof(pastix_complex64_t));
	cudaMemcpy(ValuesGPU, spm->valuesDouble, spm->nnzexp * sizeof(pastix_complex64_t), cudaMemcpyHostToDevice);
	
	pastix_int_t* colptrGPU;
	cudaMalloc((void**) &colptrGPU, (spm->n+1) * sizeof(pastix_int_t));
	cudaMemcpy(colptrGPU, spm->colptr, (spm->n+1) * sizeof(pastix_int_t), cudaMemcpyHostToDevice);
	
	pastix_int_t* rowptrGPU;
	cudaMalloc((void**) &rowptrGPU, spm->nnzexp * sizeof(pastix_int_t));
	cudaMemcpy(rowptrGPU, spm->rowptr, spm->nnzexp * sizeof(pastix_int_t), cudaMemcpyHostToDevice);
	
	pastix_int_t* permGPU;
	cudaMalloc((void**) &permGPU, spm->n * sizeof(pastix_int_t));
	cudaMemcpy(permGPU, pastix_data->ordemesh->permtab, spm->n * sizeof(pastix_int_t), cudaMemcpyHostToDevice);
	
	
	
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
	
    normb = solver.norm( pastix_data, n, d_b );
    normx = solver.norm( pastix_data, n, d_x );

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
        solver.copy( pastix_data, n, d_b, gmVi );
        
        if ( normx > 0. ) {
            //solver.spmv( pastix_data, PastixNoTrans, -1.0, d_x, 1.0, gmVi, streams);
            solver.unblocked_spmv_perm( n, -1., 1., ValuesGPU, d_x, gmVi, colptrGPU, rowptrGPU, permGPU );
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
        
		cudaMemcpy(gmG, &resid, sizeof(pastix_complex64_t), cudaMemcpyHostToDevice);
        //gmG[0] = (pastix_complex64_t)resid;
        inflag = 1;
        i = -1;
        gmHi = gmH - im1;
        gmWi = gmW - ldw;
        gmWi_host = gmW_host - ldw;
        
        while( inflag )
        {
            clockStop( refine_clk );
            t0 = clockGet();
            
            i++;

            /* Set H and W pointers to the beginning of columns i */
            gmHi = gmHi + im1;
            gmWi = gmWi + ldw;
            gmWi_host = gmWi_host + ldw;
            
            /* Backup v_{i} into w_{i} for the end */
            //solver.copy( pastix_data, n, gmVi, gmWi);
           
            /* Compute w_{i} = M^{-1} v_{i} */
            cudaMemcpy(gmWi_host, gmVi, n *  sizeof(pastix_complex64_t), cudaMemcpyDeviceToHost);
            if ( precond ) {
                solver.spsv( pastix_data, gmWi_host );
            }
			cudaMemcpy(gmWi, gmWi_host, n *  sizeof(pastix_complex64_t), cudaMemcpyHostToDevice);

            /* v_{i+1} = A (M^{-1} v_{i}) = A w_{i} */
            gmVi += n;
                                    
            //solver.unblocked_spmv( n, 1.0, 0., ValuesGPU, gmWi, gmVi, colptrGPU, rowptrGPU);
            
            //solver.spmv( pastix_data, PastixNoTrans, 1.0, gmWi, 0., gmVi, streams);
            solver.unblocked_spmv_perm( n, 1.0, 0., ValuesGPU, gmWi, gmVi, colptrGPU, rowptrGPU, permGPU);
            
            /* Classical Gram-Schmidt */
            for (j=0; j<=i; j++)
            {
                /* Compute h_{j,i} = < v_{i+1}, v_{j} > */
                solver.dot( pastix_data, n, gmVi, gmV + j * n, ((pastix_complex64_t*)gmHi)+j );
                
                /* Compute v_{i+1} = v_{i+1} - h_{j,i} v_{j} */
                cudaMemcpy(&buffer, gmHi + j, sizeof(pastix_complex64_t), cudaMemcpyDeviceToHost);
                buffer = -buffer;
                solver.axpy( pastix_data, n, buffer ,  gmV + j * n, gmVi );
                
            }
            

            /* Compute || v_{i+1} ||_f */
            norm = solver.norm( pastix_data, n, gmVi );
            
			cudaMemcpy(gmHi+i+1, &norm, sizeof(double), cudaMemcpyHostToDevice);
           //gmHi[i+1] = norm;
           

            /* Compute v_{i+1} = v_{i+1} / h_{i+1,i} iff h_{i+1,i} is not too small */
            if ( norm > 1e-50 )
            {
                tmp = (pastix_complex64_t)(1.0 / norm);
                solver.scal( pastix_data, n, tmp, gmVi );
            }

			cublasSetPointerMode(*(pastix_data->cublas_handle), CUBLAS_POINTER_MODE_DEVICE);
            /* Apply the previous Givens rotation to the new column (should call LAPACKE_zrot_work())*/
            
#if defined(PRECISION_z) ||defined(PRECISION_d)
            for (j=0; j<i;j++)
            {
				cublasZrot(*(pastix_data->cublas_handle), 1, gmHi+j, 1, gmHi+j+1, 1, (double*)(gmcos+j), gmsin+j);
            }
#else
            for (j=0; j<i;j++)
            {
				cublasZrot(*(pastix_data->cublas_handle), 1, gmHi+j, 1, gmHi+j+1, 1, (float*)(gmcos+j), gmsin+j);
            }
#endif
            /*for (j=0; j<i;j++)
            {
                tmp = gmHi[j];
                gmHi[j]   = gmcos[j] * tmp       +      gmsin[j]  * gmHi[j+1];
                gmHi[j+1] = gmcos[j] * gmHi[j+1] - conj(gmsin[j]) * tmp;
            }*/
            

            /*
             * Compute the new Givens rotation (zrotg)
             *
             * t = sqrt( h_{i,i}^2 + h_{i+1,i}^2 )
             * cos = h_{i,i}   / t
             * sin = h_{i+1,i} / t
             */
            {
				//cudaMemcpy(&bufferG1, gmHi+i, sizeof(pastix_complex64_t), cudaMemcpyDeviceToDevice);
				//cublasZrotg(*(pastix_data->cublas_handle), bufferG1, bufferG2; 
#if defined(PRECISION_z) ||defined(PRECISION_d)
            cublasZrotg(*(pastix_data->cublas_handle), gmHi+i, gmHi+i+1, (double*)(gmcos+i), gmsin+i); 
#else
            cublasZrotg(*(pastix_data->cublas_handle), gmHi+i, gmHi+i+1, (float*)(gmcos+i), gmsin+i); 
#endif 

				
				
                /*tmp = csqrt( gmHi[i]   * gmHi[i] +
                             gmHi[i+1] * gmHi[i+1] );

                if ( cabs(tmp) <= eps ) {
                    tmp = (pastix_complex64_t)eps;
                }
                gmcos[i] = gmHi[i]   / tmp;
                gmsin[i] = gmHi[i+1] / tmp;*/
            }


            /* Update the residuals (See p. 168, eq 6.35) */
            cudaMemcpy(gmG+i+1, gmG+i, sizeof(pastix_complex64_t), cudaMemcpyDeviceToDevice);
#if defined(PRECISION_z) ||defined(PRECISION_c)
            cuDoubleComplex negone = make_cuDoubleComplex(-1.0,0);
#else
            cuDoubleComplex negone = -1.0;
#endif
            cublasZscal(*(pastix_data->cublas_handle), 1, gmsin+i, gmG+i+1, 1);
            cublasZscal(*(pastix_data->cublas_handle), 1, gmcos+i, gmG+i, 1);
			cublasSetPointerMode(*(pastix_data->cublas_handle), CUBLAS_POINTER_MODE_HOST);
            cublasZscal(*(pastix_data->cublas_handle), 1, &negone, gmG+i+1, 1);
            //gmG[i+1] = -gmsin[i] * gmG[i];
            //gmG[i]   =  gmcos[i] * gmG[i];


            /* Apply the last Givens rotation */
           // gmHi[i] = gmcos[i] * gmHi[i] + gmsin[i] * gmHi[i+1];


            /* (See p. 169, eq 6.42) */
            
			
			
			pastix_complex64_t tmp_resid;
			
			cudaMemcpy(&tmp_resid, gmG+i+1, sizeof(pastix_complex64_t), cudaMemcpyDeviceToHost);
			
            resid = cabs( tmp_resid );


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

#if defined(PASTIX_DEBUG_GMRES)
                {
                    double normr2;

                    /* Compute y_m = H_m^{-1} g_m (See p. 169) */
                    memcpy( dbg_G, gmG, im1 * sizeof(pastix_complex64_t) );
                    cblas_ztrsv( CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                                 i+1, gmH, im1, dbg_G, 1 );

                    solver.copy( pastix_data, n, b, dbg_r );
                    solver.copy( pastix_data, n, x, dbg_x );

                    /* Accumulate the current v_m */
                    solver.gemv( pastix_data, n, i+1, 1.0, (precond ? gmW : gmV), n, dbg_G, 1.0, dbg_x );

        
        cudaDeviceSynchronize();
                    /* Compute b - Ax */
                    solver.unblocked_spmv( pastix_data, PastixNoTrans, -1., dbg_x, 1., dbg_r, streams );
        
        cudaDeviceSynchronize();

                    normr2 = solver.norm( pastix_data, n, dbg_r );
                    fprintf(stdout, OUT_ITERREFINE_ERR, normr2 / normb );
                }
#endif
            }
        }

        /* Compute y_m = H_m^{-1} g_m (See p. 169) */
        cublasZtrsv(*(pastix_data->cublas_handle), CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, i+1, gmH, im1, gmG, 1);
        /*cblas_ztrsv( CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                     i+1, gmH, im1, gmG, 1 );*/

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
            gmWi_host = gmW_host;
            
            solver.gemv( pastix_data, n, i+1, 1.0, gmWi, n, (pastix_complex64_t*)gmG, 1.0, d_x );
            
        }

        if ((resid_b <= eps) || (iters >= itermax))
        {
            outflag = 0;
        }
    }
    
	cudaMemcpy(x, d_x, n * sizeof(pastix_complex64_t), cudaMemcpyDeviceToHost);

    clockStop( refine_clk );
    t3 = clockGet();

    solver.output_final( pastix_data, resid_b, iters, t3, x, x );

    cudaFree(gmcos);
    cudaFree(gmsin);
    cudaFree(gmG);
    cudaFree(gmH);
    cudaFree(gmV);
    cudaFree(gmW);
#if defined(PASTIX_DEBUG_GMRES)
    cudaFree(dbg_x);
    cudaFree(dbg_r);
    cudaFree(dbg_G);
#endif

    return iters;
}
