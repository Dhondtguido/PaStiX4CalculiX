/**
 *
 * @file core_zlrothu.c
 *
 * PaStiX low-rank kernel routines to othogonalize the U matrix with QR approximations.
 *
 * @copyright 2016-2018 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @version 6.0.1
 * @author Alfredo Buttari
 * @author Gregoire Pichon
 * @date 2018-07-16
 * @precisions normal z -> c d s
 *
 **/
#include "common.h"
#include <cblas.h>
#include <lapacke.h>
#include "flops.h"
#include "kernels_trace.h"
#include "blend/solver.h"
#include "pastix_zcores.h"
#include "pastix_zlrcores.h"
#include "z_nan_check.h"
#include "pastix_lowrank.h"

#ifndef DOXYGEN_SHOULD_SKIP_THIS
static pastix_complex64_t mzone = -1.0;
static pastix_complex64_t zone  =  1.0;
static pastix_complex64_t zzero =  0.0;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/**
 *******************************************************************************
 *
 * @brief Try to orthognalize the u part of the low-rank form, and update the v
 * part accordingly using full QR.
 *
 * This function considers a low-rank matrix resulting from the addition of two
 * matrices B += A, with A of smaller or equal size to B.
 * The product has the form: U * V^t
 *
 * The U part of the low-rank form must be orthognalized to get the smaller
 * possible rank during the rradd operation. This function perfoms this by
 * applying a full QR factorization on the U part.
 *
 *  U = Q R, then U' = Q, and V' = R * V
 *
 *******************************************************************************
 *
 * @param[in] M
 *          The number of rows of the u1u2 matrix.
 *
 * @param[in] N
 *          The number of columns of the v1v2 matrix.
 *
 * @param[in] rank
 *          The number of columns of the U matrix, and the number of rows of the
 *          V part in the v1v2 matrix.
 *
 * @param[inout] U
 *          The U matrix of size ldu -by- rank. On exit, Q from U = Q R.
 *
 * @param[in] ldu
 *          The leading dimension of the U matrix. ldu >= max(1, M)
 *
 * @param[inout] V
 *          The V matrix of size ldv -by- N.
 *          On exit, R * V, with R from U = Q R.
 *
 * @param[in] ldv
 *          The leading dimension of the V matrix. ldv >= max(1, rank)
 *
 *******************************************************************************
 *
 * @return The number of flops required to perform the operation.
 *
 *******************************************************************************/
pastix_fixdbl_t
core_zlrorthu_fullqr( pastix_int_t M,  pastix_int_t N, pastix_int_t rank,
                      pastix_complex64_t *U, pastix_int_t ldu,
                      pastix_complex64_t *V, pastix_int_t ldv )
{
    pastix_int_t minMK = pastix_imin( M, rank );
    pastix_int_t lwork = M * 32 + minMK;
    pastix_int_t ret;
    pastix_complex64_t *W = malloc( lwork * sizeof(pastix_complex64_t) );
    pastix_complex64_t *tau, *work;
    pastix_fixdbl_t flops = 0.;

    tau  = W;
    work = W + minMK;
    lwork -= minMK;

    assert( M >= rank );

    /* Compute U = Q * R */
    ret = LAPACKE_zgeqrf_work( LAPACK_COL_MAJOR, M, rank,
                               U, ldu, tau, work, lwork );
    assert( ret == 0 );
    flops += FLOPS_ZGEQRF( M, rank );

    /* Compute V' = R * V' */
    cblas_ztrmm( CblasColMajor,
                 CblasLeft, CblasUpper,
                 CblasNoTrans, CblasNonUnit,
                 rank, N, CBLAS_SADDR(zone),
                 U, ldu, V, ldv );
    flops += FLOPS_ZTRMM( PastixLeft, rank, N );

    /* Generate the Q */
    ret = LAPACKE_zungqr_work( LAPACK_COL_MAJOR, M, rank, rank,
                               U, ldu, tau, work, lwork );
    assert( ret == 0 );
    flops += FLOPS_ZUNGQR( M, rank, rank );

    free(W);

    (void)ret;
    return flops;
}

/**
 *******************************************************************************
 *
 * @brief Try to orthognalize the U part of the low-rank form, and update the V
 * part accordingly using partial QR.
 *
 * This function considers a low-rank matrix resulting from the addition of two
 * matrices B += A, with A of smaller or equal size to B.
 * The product has the form: U * V^t
 *
 * The U part of the low-rank form must be orthognalized to get the smaller
 * possible rank during the rradd operation. This function perfoms this by
 * applying a full QR factorization on the U part.
 *
 * In that case, it takes benefit from the fact that U = [ u1, u2 ], and V = [
 * v1, v2 ] with u2 and v2 wich are matrices of respective size M2-by-r2, and
 * r2-by-N2, offset by offx and offy
 *
 * The steps are:
 *    - Scaling of u2 with removal of the null columns
 *    - Orthogonalization of u2 relatively to u1
 *    - Application of the update to v2
 *    - orthogonalization through QR of u2
 *    - Update of V
 *
 *******************************************************************************
 *
 * @param[in] M
 *          The number of rows of the u1u2 matrix.
 *
 * @param[in] N
 *          The number of columns of the v1v2 matrix.
 *
 * @param[in] r1
 *          The number of columns of the U matrix in the u1 part, and the number
 *          of rows of the V part in the v1 part.
 *
 * @param[inout] r2ptr
 *          The number of columns of the U matrix in the u2 part, and the number
 *          of rows of the V part in the v2 part. On exit, this rank is reduced
 *          y the number of null columns found in U.
 *
 * @param[in] offx
 *          The row offset of the matrix u2 in U.
 *
 * @param[in] offy
 *          The column offset of the matrix v2 in V.
 *
 * @param[inout] U
 *          The U matrix of size ldu -by- rank. On exit, the orthogonalized U.
 *
 * @param[in] ldu
 *          The leading dimension of the U matrix. ldu >= max(1, M)
 *
 * @param[inout] V
 *          The V matrix of size ldv -by- N.
 *          On exit, the updated V matrix.
 *
 * @param[in] ldv
 *          The leading dimension of the V matrix. ldv >= max(1, rank)
 *
 *******************************************************************************
 *
 * @return The number of flops required to perform the operation.
 *
 *******************************************************************************/
pastix_fixdbl_t
core_zlrorthu_partialqr( pastix_int_t M,  pastix_int_t N,
                         pastix_int_t r1, pastix_int_t *r2ptr,
                         pastix_int_t offx, pastix_int_t offy,
                         pastix_complex64_t *U, pastix_int_t ldu,
                         pastix_complex64_t *V, pastix_int_t ldv )
{
    pastix_int_t r2 = *r2ptr;
    pastix_int_t minMN = pastix_imin( M, r2 );
    pastix_int_t ldwork = pastix_imax( r1 * r2, M * 32 + minMN );
    pastix_int_t ret, i;
    pastix_complex64_t *u1 = U;
    pastix_complex64_t *u2 = U + r1 * ldu;
    pastix_complex64_t *v1 = V;
    pastix_complex64_t *v2 = V + r1;
    pastix_complex64_t *W = malloc( ldwork * sizeof(pastix_complex64_t) );
    pastix_complex64_t *tau, *work;
    pastix_fixdbl_t flops = 0.;
    double norm, eps;

    tau = W;
    work = W + minMN;
    ldwork -= minMN;

    eps = LAPACKE_dlamch_work('e');

    /* Scaling */
    for (i=0; i<r2; i++, u2 += ldu, v2++) {
        norm = cblas_dznrm2( M, u2, 1 );
        if ( norm > (M * eps) ) {
            cblas_zdscal( M, 1. / norm, u2, 1   );
            cblas_zdscal( N, norm,      v2, ldv );
        }
        else {
            if ( i < (r2-1) ) {
                cblas_zswap( M, u2, 1, U + (r1+r2-1) * ldu, 1 );
                memset( U + (r1+r2-1) * ldu, 0,  M * sizeof(pastix_complex64_t) );

                cblas_zswap( N, v2, ldv, V + (r1+r2-1),     ldv );
                LAPACKE_zlaset_work( LAPACK_COL_MAJOR, 'A', 1, N,
                                     0., 0., V + (r1+r2-1), ldv );
                r2--;
                i--;
                u2-= ldu;
                v2--;
            }
            else {
                memset( u2, 0,  M * sizeof(pastix_complex64_t) );
                LAPACKE_zlaset_work( LAPACK_COL_MAJOR, 'A', 1, N,
                                     0., 0., v2, ldv );
                r2--;
            }
        }
    }
    u2 = U + r1 * ldu;
    v2 = V + r1;

    *r2ptr = r2;

    if ( r2 == 0 ) {
        free( W );
        return 0.;
    }

    /* Compute W = u1^t u2 */
    cblas_zgemm( CblasColMajor, CblasConjTrans, CblasNoTrans,
                 r1, r2, M,
                 CBLAS_SADDR(zone),  u1, ldu,
                                     u2, ldu,
                 CBLAS_SADDR(zzero), W,  r1 );
    flops += FLOPS_ZGEMM( r1, r2, M );

    /* Compute u2 = u2 - u1 ( u1^t u2 ) = u2 - u1 * W */
    cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                 M, r2, r1,
                 CBLAS_SADDR(mzone), u1, ldu,
                                     W,  r1,
                 CBLAS_SADDR(zone),  u2, ldu );
    flops += FLOPS_ZGEMM( M, r2, r1 );

    /* Update v1 = v1 + ( u1^t u2 ) v2 = v1 + W * v2 */
    cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                 r1, N, r2,
                 CBLAS_SADDR(zone), W,  r1,
                                    v2, ldv,
                 CBLAS_SADDR(zone), v1, ldv );
    flops += FLOPS_ZGEMM( r1, N, r2 );

#if !defined(PASTIX_LR_CGS1)
    /* Compute W = u1^t u2 */
    cblas_zgemm( CblasColMajor, CblasConjTrans, CblasNoTrans,
                 r1, r2, M,
                 CBLAS_SADDR(zone),  u1, ldu,
                                     u2, ldu,
                 CBLAS_SADDR(zzero), W,  r1 );
    flops += FLOPS_ZGEMM( r1, r2, M );

    /* Compute u2 = u2 - u1 ( u1^t u2 ) = u2 - u1 * W */
    cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                 M, r2, r1,
                 CBLAS_SADDR(mzone), u1, ldu,
                                     W,  r1,
                 CBLAS_SADDR(zone),  u2, ldu );
    flops += FLOPS_ZGEMM( M, r2, r1 );

    /* Update v1 = v1 + ( u1^t u2 ) v2 = v1 + W * v2 */
    cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                 r1, N, r2,
                 CBLAS_SADDR(zone), W,  r1,
                                    v2, ldv,
                 CBLAS_SADDR(zone), v1, ldv );
    flops += FLOPS_ZGEMM( r1, N, r2 );
#endif

#if defined(PASTIX_DEBUG_LR)
    if ( core_zlrdbg_check_orthogonality_AB( M, r1, r2, u1, ldu, u2, ldu ) != 0 ) {
        fprintf(stderr, "partialQR: u2 not correctly projected with u1\n" );
    }
#endif

    /* Compute u2 = Q * R */
    ret = LAPACKE_zgeqrf_work( LAPACK_COL_MAJOR, M, r2,
                               u2, ldu, tau, work, ldwork );
    assert( ret == 0 );
    flops += FLOPS_ZGEQRF( M, r2 );

    /* Compute v2' = R * v2 */
    cblas_ztrmm( CblasColMajor,
                 CblasLeft, CblasUpper,
                 CblasNoTrans, CblasNonUnit,
                 r2, N, CBLAS_SADDR(zone),
                 u2, ldu, v2, ldv);
    flops += FLOPS_ZTRMM( PastixLeft, r2, N );

    /* Generate the Q */
    ret = LAPACKE_zungqr_work( LAPACK_COL_MAJOR, M, r2, r2,
                               u2, ldu, tau, work, ldwork );
    assert( ret == 0 );
    flops += FLOPS_ZUNGQR( M, r2, r2 );

#if defined(PASTIX_DEBUG_LR)
    if ( core_zlrdbg_check_orthogonality_AB( M, r1, r2, u1, ldu, u2, ldu ) != 0 ) {
        fprintf(stderr, "partialQR: Final u2 not orthogonal to u1\n" );
    }
#endif

    free( W );

    (void)ret;
    (void)offx;
    (void)offy;

    return flops;
}

/**
 *******************************************************************************
 *
 * @brief Try to orthognalize the U part of the low-rank form, and update the V
 * part accordingly using CGS.
 *
 * This function considers a low-rank matrix resulting from the addition of two
 * matrices B += A, with A of smaller or equal size to B.
 * The product has the form: U * V^t
 *
 * The U part of the low-rank form must be orthognalized to get the smaller
 * possible rank during the rradd operation. This function perfoms this by
 * applying a full QR factorization on the U part.
 *
 * In that case, it takes benefit from the fact that U = [ u1, u2 ], and V = [
 * v1, v2 ] with u2 and v2 wich are matrices of respective size M2-by-r2, and
 * r2-by-N2, offset by offx and offy
 *
 * The steps are:
 *    - for each column of u2
 *       - Scaling of u2 with removal of the null columns
 *       - Orthogonalization of u2 relatively to u1
 *       - Remove the column if null
 *
 *******************************************************************************
 *
 * @param[in] M1
 *          The number of rows of the U matrix.
 *
 * @param[in] N1
 *          The number of columns of the U matrix.
 *
 * @param[in] M2
 *          The number of rows of the u2 part of the U matrix.
 *
 * @param[in] N2
 *          The number of columns of the v2 part of the V matrix.
 *
 * @param[in] r1
 *          The number of columns of the U matrix in the u1 part, and the number
 *          of rows of the V part in the v1 part.
 *
 * @param[inout] r2ptr
 *          The number of columns of the U matrix in the u2 part, and the number
 *          of rows of the V part in the v2 part. On exit, this rank is reduced
 *          y the number of null columns found in U.
 *
 * @param[in] offx
 *          The row offset of the matrix u2 in U.
 *
 * @param[in] offy
 *          The column offset of the matrix v2 in V.
 *
 * @param[inout] U
 *          The U matrix of size ldu -by- rank. On exit, the orthogonalized U.
 *
 * @param[in] ldu
 *          The leading dimension of the U matrix. ldu >= max(1, M)
 *
 * @param[inout] V
 *          The V matrix of size ldv -by- N.
 *          On exit, the updated V matrix.
 *
 * @param[in] ldv
 *          The leading dimension of the V matrix. ldv >= max(1, rank)
 *
 *******************************************************************************
 *
 * @return The number of flops required to perform the operation.
 *
 *******************************************************************************/
pastix_fixdbl_t
core_zlrorthu_cgs( pastix_int_t M1,  pastix_int_t N1,
                   pastix_int_t M2,  pastix_int_t N2,
                   pastix_int_t r1, pastix_int_t *r2ptr,
                   pastix_int_t offx, pastix_int_t offy,
                   pastix_complex64_t *U, pastix_int_t ldu,
                   pastix_complex64_t *V, pastix_int_t ldv )
{
    pastix_int_t r2 = *r2ptr;
    pastix_complex64_t *u1 = U;
    pastix_complex64_t *u2 = U + r1 * ldu;
    pastix_complex64_t *v1 = V;
    pastix_complex64_t *v2 = V + r1;
    pastix_complex64_t *W;
    pastix_fixdbl_t flops = 0.0;
    pastix_int_t i, rank = r1 + r2;
    pastix_int_t ldwork = rank;
    double eps, norm;
    double norm_before, alpha;

    assert( M1 >= (M2 + offx) );
    assert( N1 >= (N2 + offy) );

    W     = malloc(ldwork * sizeof(pastix_complex64_t));
    eps   = LAPACKE_dlamch( 'e' );
    alpha = 1. / sqrt(2);

    /* Classical Gram-Schmidt */
    for (i=r1; i<rank; i++, u2 += ldu, v2++) {

        norm = cblas_dznrm2( M2, u2 + offx, 1 );
        if ( norm > ( M2 * eps ) ) {
            cblas_zdscal( M2, 1. / norm, u2 + offx,       1   );
            cblas_zdscal( N2, norm,      v2 + offy * ldv, ldv );
        }
        else {
            rank--; r2--;
            if ( i < rank ) {
                cblas_zswap( M2, u2 + offx, 1, U + rank * ldu + offx, 1 );
#if !defined(NDEBUG)
                memset( U + rank * ldu, 0,  M1 * sizeof(pastix_complex64_t) );
#endif

                cblas_zswap( N2, v2 + offy * ldv, ldv, V + offy * ldv + rank, ldv );

#if !defined(NDEBUG)
                LAPACKE_zlaset_work( LAPACK_COL_MAJOR, 'A', 1, N1,
                                     0., 0., V + rank, ldv );
#endif
                i--;
                u2-= ldu;
                v2--;
            }
#if !defined(NDEBUG)
            else {
                memset( u2, 0,  M1 * sizeof(pastix_complex64_t) );
                LAPACKE_zlaset_work( LAPACK_COL_MAJOR, 'A', 1, N1,
                                     0., 0., v2, ldv );
            }
#endif
            continue;
        }

        /* Compute W = u1^t u2 */
        cblas_zgemv( CblasColMajor, CblasConjTrans,
                     M2, i,
                     CBLAS_SADDR(zone),  u1+offx, ldu,
                                         u2+offx, 1,
                     CBLAS_SADDR(zzero), W,       1 );
        flops += FLOPS_ZGEMM( M2, i, 1 );

        /* Compute u2 = u2 - u1 ( u1^t u2 ) = u2 - u1 * W */
        cblas_zgemv( CblasColMajor, CblasNoTrans,
                     M1, i,
                     CBLAS_SADDR(mzone), u1, ldu,
                                         W,  1,
                     CBLAS_SADDR(zone),  u2, 1 );
        flops += FLOPS_ZGEMM( M1, i, 1 );

        /* Update v1 = v1 + ( u1^t u2 ) v2 = v1 + W * v2 */
        cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                     i, N1, 1,
                     CBLAS_SADDR(zone), W,  i,
                                        v2, ldv,
                     CBLAS_SADDR(zone), v1, ldv );
        flops += FLOPS_ZGEMM( i, N1, 1 );

        norm_before = cblas_dznrm2( i, W,  1 );
        norm        = cblas_dznrm2( M1, u2, 1 );

#if !defined(PASTIX_LR_CGS1)
        if ( norm <= (alpha * norm_before) ){
            /* Compute W = u1^t u2 */
            cblas_zgemv( CblasColMajor, CblasConjTrans,
                         M1, i,
                         CBLAS_SADDR(zone),  u1, ldu,
                                             u2, 1,
                         CBLAS_SADDR(zzero), W,  1 );
            flops += FLOPS_ZGEMM( M1, i, 1 );

            /* Compute u2 = u2 - u1 ( u1^t u2 ) = u2 - u1 * W */
            cblas_zgemv( CblasColMajor, CblasNoTrans,
                         M1, i,
                         CBLAS_SADDR(mzone), u1, ldu,
                                             W,  1,
                         CBLAS_SADDR(zone),  u2, 1 );
            flops += FLOPS_ZGEMM( M1, i, 1 );

            /* Update v1 = v1 + ( u1^t u2 ) v2 = v1 + W * v2 */
            cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                         i, N1, 1,
                         CBLAS_SADDR(zone), W,  i,
                                            v2, ldv,
                         CBLAS_SADDR(zone), v1, ldv );
            flops += FLOPS_ZGEMM( i, N1, 1 );

            norm = cblas_dznrm2( M1, u2, 1 );
        }
#endif

        if ( norm > M1 * eps ) {
            cblas_zdscal( M1, 1. / norm, u2, 1   );
            cblas_zdscal( N1, norm,      v2, ldv );
        }
        else {
            rank--; r2--;
            if ( i < rank ) {
                cblas_zswap( M1, u2, 1, U + rank * ldu, 1 );
                memset( U + rank * ldu, 0,  M1 * sizeof(pastix_complex64_t) );

                cblas_zswap( N1, v2, ldv, V + rank,     ldv );
                LAPACKE_zlaset_work( LAPACK_COL_MAJOR, 'A', 1, N1,
                                     0., 0., V + rank, ldv );
                i--;
                u2-= ldu;
                v2--;
            }
            else {
                memset( u2, 0,  M1 * sizeof(pastix_complex64_t) );
                LAPACKE_zlaset_work( LAPACK_COL_MAJOR, 'A', 1, N1,
                                     0., 0., v2, ldv );
            }
        }
    }
    free(W);

#if defined(PASTIX_DEBUG_LR)
    {
        u2 = U + r1 * ldu;
        if ( core_zlrdbg_check_orthogonality_AB( M1, r1, r2, u1, ldu, u2, ldu ) != 0 ) {
            fprintf(stderr, "cgs: Final u2 not orthogonal to u1\n" );
        }
    }
#endif

    *r2ptr = r2;

    (void)offy;
    (void)N2;
    return flops;
}
