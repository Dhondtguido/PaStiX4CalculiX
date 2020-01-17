/**
 * @file bcsc_z.h
 *
 * @copyright 2004-2018 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @version 6.0.1
 * @author Mathieu Faverge
 * @author Pierre Ramet
 * @author Xavier Lacoste
 * @date 2018-07-16
 *
 * @precisions normal z -> c d s
 *
 **/
#ifndef _bcsc_z_h_
#define _bcsc_z_h_



void transpose_z_Matrix(pastix_int_t 		n,
						pastix_int_t* 		colptrIn, 
						pastix_int_t* 		rowptrIn, 
						cuDoubleComplex*	valuesIn, 
						cuDoubleComplex* 	valuesOut);

void permute_z_Matrix(pastix_int_t		n,
					  pastix_int_t* 	colptrIn,
					  pastix_int_t* 	rowptrIn, 
					  cuDoubleComplex* 	valuesIn, 
					  pastix_int_t* 	perm, 
					  pastix_int_t* 	peri, 
					  pastix_int_t* 	colptrOut, 
					  pastix_int_t* 	rowptrOut, 
					  cuDoubleComplex*  valuesOut);
					  
/**
 * @addtogroup bcsc_internal
 * @{
 *
 *    @name PastixComplex64 initialization functions
 *    @{
 */
void bcsc_zinit_centralized( const spmatrix_t     *spm,
                             const pastix_order_t *ord,
                             const SolverMatrix   *solvmtx,
                             const pastix_int_t   *col2cblk,
                                   int             initAt,
                                   pastix_bcsc_t  *bcsc );
                                   
void bcsc_zsort( pastix_bcsc_t *bcsc,
            pastix_int_t        **rowtab,
            pastix_complex64_t  **valtab,
            pastix_int_t  		**sorttab);
            
/**
 *   @}
 * @}
 *
 * @addtogroup bcsc
 * @{
 *
 *    @name PastixComplex64 vector(s) operations
 *    @{
 */
void bvec_zaxpy_seq( pastix_data_t            *pastix_data,
                     pastix_int_t              n,
                     pastix_complex64_t        alpha,
                     const pastix_complex64_t *x,
                     pastix_complex64_t       *y );
void bvec_zaxpy_smp( pastix_data_t            *pastix_data,
                     pastix_int_t              n,
                     pastix_complex64_t        alpha,
                     const pastix_complex64_t *x,
                     pastix_complex64_t       *y );
void bvec_zaxpy_cuda( pastix_data_t            *pastix_data,
                     pastix_int_t              n,
                     cuDoubleComplex        alpha,
                     const cuDoubleComplex *x,
                     cuDoubleComplex       *y );

void bvec_zcopy_seq( pastix_data_t            *pastix_data,
                     pastix_int_t              n,
                     const pastix_complex64_t *x,
                     pastix_complex64_t       *y );
void bvec_zcopy_smp( pastix_data_t            *pastix_data,
                     pastix_int_t              n,
                     const pastix_complex64_t *x,
                     pastix_complex64_t       *y );
void bvec_zcopy_cuda( pastix_data_t            *pastix_data,
                     pastix_int_t              n,
                     const cuDoubleComplex *x,
                     cuDoubleComplex       *y );

#if defined(PRECISION_z) || defined(PRECISION_c)
void bvec_zdotc_seq( pastix_data_t            *pastix_data,
                                   pastix_int_t              n,
                                   const pastix_complex64_t *x,
                                   const pastix_complex64_t *y,
                                         pastix_complex64_t *r  );
void bvec_zdotc_smp( pastix_data_t            *pastix_data,
                                   pastix_int_t              n,
                                   const pastix_complex64_t *x,
                                   const pastix_complex64_t *y,
                                         pastix_complex64_t *r  );
void bvec_zdotc_cuda( pastix_data_t            *pastix_data,
                                   pastix_int_t              n,
                                   const cuDoubleComplex *x,
                                   const cuDoubleComplex *y,
                                         cuDoubleComplex *r  );
#endif

void bvec_zdotu_seq( pastix_data_t            *pastix_data,
                                   pastix_int_t              n,
                                   const pastix_complex64_t *x,
                                   const pastix_complex64_t *y,
                                         pastix_complex64_t *r  );
void bvec_zdotu_smp( pastix_data_t            *pastix_data,
                                   pastix_int_t              n,
                                   const pastix_complex64_t *x,
                                   const pastix_complex64_t *y,
                                         pastix_complex64_t *r  );
void bvec_zdotu_cuda( pastix_data_t            *pastix_data,
                                   pastix_int_t              n,
                                   const cuDoubleComplex *x,
                                   const cuDoubleComplex *y,
                                         cuDoubleComplex *r  );

void bvec_zgemv_seq( pastix_data_t            *pastix_data,
                     pastix_int_t              m,
                     pastix_int_t              n,
                     pastix_complex64_t        alpha,
                     const pastix_complex64_t *A,
                     pastix_int_t              lda,
                     const pastix_complex64_t *x,
                     pastix_complex64_t        beta,
                     pastix_complex64_t       *y );
void bvec_zgemv_smp( pastix_data_t            *pastix_data,
                     pastix_int_t              m,
                     pastix_int_t              n,
                     pastix_complex64_t        alpha,
                     const pastix_complex64_t *A,
                     pastix_int_t              lda,
                     const pastix_complex64_t *x,
                     pastix_complex64_t        beta,
                     pastix_complex64_t       *y );
void bvec_zgemv_cuda( pastix_data_t            *pastix_data,
                     pastix_int_t              m,
                     pastix_int_t              n,
                     cuDoubleComplex        alpha,
                     const cuDoubleComplex *A,
                     pastix_int_t              lda,
                     const cuDoubleComplex *x,
                     cuDoubleComplex        beta,
                     cuDoubleComplex       *y );

double bvec_znrm2_seq( pastix_data_t            *pastix_data,
                       pastix_int_t              n,
                       const pastix_complex64_t *x );
double bvec_znrm2_smp( pastix_data_t            *pastix_data,
                       pastix_int_t              n,
                       const pastix_complex64_t *x );
double bvec_znrm2_cuda( pastix_data_t            *pastix_data,
                       pastix_int_t              n,
                       const cuDoubleComplex *x );

void bvec_zscal_seq( pastix_data_t      *pastix_data,
                     pastix_int_t        n,
                     pastix_complex64_t  alpha,
                     pastix_complex64_t *x );
void bvec_zscal_smp( pastix_data_t      *pastix_data,
                     pastix_int_t        n,
                     pastix_complex64_t  alpha,
                     pastix_complex64_t *x );
void bvec_zscal_cuda( pastix_data_t      *pastix_data,
                     pastix_int_t        n,
                     cuDoubleComplex  alpha,
                     cuDoubleComplex *x );

int bvec_zlapmr( int thread_safe,
                 pastix_dir_t        dir,
                 pastix_int_t        m,
                 pastix_int_t        n,
                 pastix_complex64_t *A,
                 pastix_int_t        lda,
                 pastix_int_t       *perm );

/**
 *    @}
 *
 *    @name PastixComplex64 matrix operations
 *    @{
 */
double bcsc_znorm( pastix_normtype_t    ntype,
                   const pastix_bcsc_t *bcsc );

void bcsc_zspsv( pastix_data_t      *pastix_data,
                 pastix_complex64_t *b );

void bcsc_zspmv( const pastix_data_t      *pastix_data,
                 pastix_trans_t            trans,
                 pastix_complex64_t        alpha,
                 const pastix_complex64_t *x,
                 pastix_complex64_t        beta,
                 pastix_complex64_t       *y,
				 cudaStream_t		        *streams  );

void bcsc_zspmv_seq( const pastix_data_t      *pastix_data,
                     pastix_trans_t            trans,
                     pastix_complex64_t        alpha,
                     const pastix_complex64_t *x,
                     pastix_complex64_t        beta,
                     pastix_complex64_t       *y );
void bcsc_zspmv_smp( const pastix_data_t      *pastix_data,
                     pastix_trans_t            trans,
                     pastix_complex64_t        alpha,
                     const pastix_complex64_t *x,
                     pastix_complex64_t        beta,
                     pastix_complex64_t       *y );
void bcsc_zspmv_cuda( const pastix_data_t      *pastix_data,
                     pastix_trans_t            trans,
                     pastix_complex64_t        alpha,
                     const pastix_complex64_t *x,
                     pastix_complex64_t        beta,
                     pastix_complex64_t       *y,
				     cudaStream_t		        *streams );

/**
 *    @}
 * @}
 */
#endif /* _bcsc_z_h_ */
