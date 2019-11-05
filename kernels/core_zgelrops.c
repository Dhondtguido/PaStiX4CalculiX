/**
 *
 * @file core_zgelrops.c
 *
 * PaStiX low-rank kernel routines
 *
 * @copyright 2016-2018 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @version 6.0.1
 * @author Gregoire Pichon
 * @date 2018-07-16
 * @precisions normal z -> c d s
 *
 **/
#include "common.h"
#include <cblas.h>
#include <lapacke.h>
#include "blend/solver.h"
#include "pastix_zcores.h"
#include "pastix_zlrcores.h"
#include "z_nan_check.h"
#include "kernels_trace.h"

#ifndef DOXYGEN_SHOULD_SKIP_THIS
static pastix_complex64_t zone  =  1.0;
static pastix_complex64_t zzero =  0.0;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/**
 *******************************************************************************
 *
 * @brief Allocate a low-rank matrix.
 *
 *******************************************************************************
 *
 * @param[in] M
 *          Number of rows of the matrix A.
 *
 * @param[in] N
 *          Number of columns of the matrix A.
 *
 * @param[in] rkmax
 *         @arg -1: the matrix is allocated tight to its rank.
 *         @arg >0: the matrix is allocated to the minimum of rkmax and its maximum rank.
 *
 * @param[out] A
 *          The allocated low-rank matrix
 *
 *******************************************************************************/
void
core_zlralloc( pastix_int_t      M,
               pastix_int_t      N,
               pastix_int_t      rkmax,
               pastix_lrblock_t *A )
{
    pastix_complex64_t *u, *v;

    if ( rkmax == -1 ) {
        u = malloc( M * N * sizeof(pastix_complex64_t) );
        memset(u, 0, M * N * sizeof(pastix_complex64_t) );
        A->rk = -1;
        A->rkmax = M;
        A->u = u;
        A->v = NULL;
    }
    else {
        pastix_int_t rk = pastix_imin( M, N );
        rkmax = pastix_imin( rkmax, rk );

#if defined(PASTIX_DEBUG_LR)
        u = malloc( M * rkmax * sizeof(pastix_complex64_t) );
        v = malloc( N * rkmax * sizeof(pastix_complex64_t) );

        /* To avoid uninitialised values in valgrind. Lapacke doc (xgesvd) is not correct */
        memset(u, 0, M * rkmax * sizeof(pastix_complex64_t));
        memset(v, 0, N * rkmax * sizeof(pastix_complex64_t));
#else
        u = malloc( (M+N) * rkmax * sizeof(pastix_complex64_t));

        /* To avoid uninitialised values in valgrind. Lapacke doc (xgesvd) is not correct */
        memset(u, 0, (M+N) * rkmax * sizeof(pastix_complex64_t));

        v = u + M * rkmax;
#endif

        A->rk = 0;
        A->rkmax = rkmax;
        A->u = u;
        A->v = v;
    }
}

/**
 *******************************************************************************
 *
 * @brief Free a low-rank matrix.
 *
 *******************************************************************************
 *
 * @param[inout] A
 *          The low-rank matrix that will be desallocated.
 *
 *******************************************************************************/
void
core_zlrfree( pastix_lrblock_t *A )
{
    if ( A->rk == -1 ) {
        free(A->u);
        A->u = NULL;
    }
    else {
        free(A->u);
#if defined(PASTIX_DEBUG_LR)
        free(A->v);
#endif
        A->u = NULL;
        A->v = NULL;
    }
    A->rk = 0;
    A->rkmax = 0;
}

/**
 *******************************************************************************
 *
 * @brief Resize a low-rank matrix
 *
 *******************************************************************************
 *
 * @param[in] copy
 *          Enable/disable the copy of the data from A->u and A->v into the new
 *          low-rank representation.
 *
 * @param[in] M
 *          The number of rows of the matrix A.
 *
 * @param[in] N
 *          The number of columns of the matrix A.
 *
 * @param[inout] A
 *          The low-rank representation of the matrix. At exit, this structure
 *          is modified with the new low-rank representation of A, is the rank
 *          is small enough
 *
 * @param[in] newrk
 *          The new rank of the matrix A.
 *
 * @param[in] newrkmax
 *          The new maximum rank of the matrix A. Useful if the low-rank
 *          structure was allocated with more data than the rank.
 *
 * @param[in] rklimit
 *          The maximum rank to store the matrix in low-rank format. If
 *          -1, set to core_get_rklimit(M, N)
 *
 *******************************************************************************
 *
 * @return  The new rank of A
 *
 *******************************************************************************/
int
core_zlrsze( int copy, pastix_int_t M, pastix_int_t N,
             pastix_lrblock_t *A,
             pastix_int_t newrk,
             pastix_int_t newrkmax,
             pastix_int_t rklimit )
{
    /* If no limit on the rank is given, let's take min(M, N) */
    rklimit = (rklimit == -1) ? core_get_rklimit( M, N ) : rklimit;

    /* If no extra memory allocated, let's fix rkmax to rk */
    newrkmax = (newrkmax == -1) ? newrk : newrkmax;

    /*
     * It is not interesting to compress, so we alloc space to store the full matrix
     */
    if ( (newrk > rklimit) || (newrk == -1) )
    {
        A->u = realloc( A->u, M * N * sizeof(pastix_complex64_t) );
#if defined(PASTIX_DEBUG_LR)
        free(A->v);
#endif
        A->v = NULL;
        A->rk = -1;
        A->rkmax = M;
        return -1;
    }
    /*
     * The rank is null, we free everything
     */
    else if (newrkmax == 0)
    {
        /*
         * The rank is null, we free everything
         */
        free(A->u);
#if defined(PASTIX_DEBUG_LR)
        free(A->v);
#endif
        A->u = NULL;
        A->v = NULL;
        A->rkmax = newrkmax;
        A->rk = newrk;
    }
    /*
     * The rank is non null, we allocate the correct amount of space, and
     * compress the stored information if necessary
     */
    else {
        pastix_complex64_t *u, *v;
        int ret;

        if ( ( A->rk == -1 ) || (newrkmax > A->rkmax) ) {
#if defined(PASTIX_DEBUG_LR)
            u = malloc( M * newrkmax * sizeof(pastix_complex64_t) );
            v = malloc( N * newrkmax * sizeof(pastix_complex64_t) );
#else
            u = malloc( (M+N) * newrkmax * sizeof(pastix_complex64_t) );
            v = u + M * newrkmax;
#endif
            if ( copy ) {
                assert( A->rk != -1 );
                ret = LAPACKE_zlacpy_work( LAPACK_COL_MAJOR, 'A', M, newrk,
                                           A->u, M, u, M );
                assert(ret == 0);
                ret = LAPACKE_zlacpy_work( LAPACK_COL_MAJOR, 'A', newrk, N,
                                           A->v, A->rkmax, v, newrkmax );
                assert(ret == 0);
            }
            free(A->u);
#if defined(PASTIX_DEBUG_LR)
            free(A->v);
#endif
            A->u = u;
            A->v = v;
        }

        /* Update rk and rkmax */
        A->rkmax = A->rk == -1 ? newrkmax : pastix_imax( newrkmax, A->rkmax );
        A->rk    = newrk;

        (void)ret;
    }
    assert( A->rk <= A->rkmax);
    return 0;
}

/**
 *******************************************************************************
 *
 * @brief Convert a low rank matrix into a dense matrix.
 *
 * Convert a low-rank matrix of size m-by-n into a full rank matrix.
 * A = op( u * v^t ) with op(A) = A or A^t
 *
 *******************************************************************************
 *
 * @param[in] trans
 *          @arg PastixNoTrans: returns A = u * v^t
 *          @arg PastixTrans: returns A = v * u^t
 *
 * @param[in] m
 *          Number of rows of the low-rank matrix Alr.
 *
 * @param[in] n
 *          Number of columns of the low-rank matrix Alr.
 *
 * @param[in] Alr
 *          The low rank matrix to be converted into a full rank matrix
 *
 * @param[inout] A
 *          The matrix of dimension lda-by-k in which to store the uncompressed
 *          version of Alr. k = n if trans == PastixNoTrans, m otherwise.
 *
 * @param[in] lda
 *          The leading dimension of the matrix A. lda >= max(1, m) if trans ==
 *          PastixNoTrans, lda >= max(1,n) otherwise.
 *
 *******************************************************************************
 *
 * @retval  0  in case of success.
 * @retval  -i if the ith parameter is incorrect.
 *
 *******************************************************************************/
int
core_zlr2ge( pastix_trans_t trans, pastix_int_t m, pastix_int_t n,
             const pastix_lrblock_t *Alr,
             pastix_complex64_t *A, pastix_int_t lda )
{
    int ret = 0;

#if !defined(NDEBUG)
    if ( m < 0 ) {
        return -1;
    }
    if ( n < 0 ) {
        return -2;
    }
    if (Alr == NULL || Alr->rk > Alr->rkmax) {
        return -3;
    }
    if ( (trans == PastixNoTrans && lda < m) ||
         (trans != PastixNoTrans && lda < n) )
    {
        return -5;
    }
    if ( Alr->rk == -1 ) {
        if (Alr->u == NULL || Alr->v != NULL || (Alr->rkmax < m))
        {
            return -6;
        }
    }
    else if ( Alr->rk != 0){
        if (Alr->u == NULL || Alr->v == NULL) {
            return -6;
        }
    }
#endif

    if ( trans == PastixNoTrans ) {
        if ( Alr->rk == -1 ) {
            ret = LAPACKE_zlacpy_work( LAPACK_COL_MAJOR, 'A', m, n,
                                       Alr->u, Alr->rkmax, A, lda );
            assert( ret == 0 );
        }
        else if ( Alr->rk == 0 ) {
            ret = LAPACKE_zlaset_work( LAPACK_COL_MAJOR, 'A', m, n,
                                       0.0, 0.0, A, lda );
            assert( ret == 0 );
        }
        else {
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        m, n, Alr->rk,
                        CBLAS_SADDR(zone),  Alr->u, m,
                                            Alr->v, Alr->rkmax,
                        CBLAS_SADDR(zzero), A, lda);
        }
    }
    else {
        if ( Alr->rk == -1 ) {
            core_zgetro( m, n, Alr->u, Alr->rkmax, A, lda );
        }
        else if ( Alr->rk == 0 ) {
            ret = LAPACKE_zlaset_work( LAPACK_COL_MAJOR, 'A', n, m,
                                       0.0, 0.0, A, lda );
            assert( ret == 0 );
        }
        else {
            cblas_zgemm(CblasColMajor, CblasTrans, CblasTrans,
                        n, m, Alr->rk,
                        CBLAS_SADDR(zone),  Alr->v, Alr->rkmax,
                                            Alr->u, m,
                        CBLAS_SADDR(zzero), A,      lda);
        }
    }

    return ret;
}

/**
 *******************************************************************************
 *
 * @brief Copy a small low-rank structure into a large one
 *
 *******************************************************************************
 *
 * @param[in] lowrank
 *          The structure with low-rank parameters.
 *
 * @param[in] transAv
 *         @arg PastixNoTrans:   (A.v)' is stored transposed as usual
 *         @arg PastixTrans:      A.v is stored
 *         @arg PastixConjTrans:  A.v is stored
 *
 * @param[in] alpha
 *          The multiplier parameter: B = B + alpha * A
 *
 * @param[in] M1
 *          The number of rows of the matrix A.
 *
 * @param[in] N1
 *          The number of columns of the matrix A.
 *
 * @param[in] A
 *          The low-rank representation of the matrix A.
 *
 * @param[in] M2
 *          The number of rows of the matrix B.
 *
 * @param[in] N2
 *          The number of columns of the matrix B.
 *
 * @param[inout] B
 *          The low-rank representation of the matrix B.
 *
 * @param[in] offx
 *          The horizontal offset of A with respect to B.
 *
 * @param[in] offy
 *          The vertical offset of A with respect to B.
 *
 *******************************************************************************/
void
core_zlrcpy( const pastix_lr_t *lowrank,
             pastix_trans_t transAv, pastix_complex64_t alpha,
             pastix_int_t M1, pastix_int_t N1, const pastix_lrblock_t *A,
             pastix_int_t M2, pastix_int_t N2,       pastix_lrblock_t *B,
             pastix_int_t offx, pastix_int_t offy )
{
    pastix_complex64_t *u, *v;
    pastix_int_t ldau, ldav;
    int ret;

    assert( A->rk <= core_get_rklimit( M2, N2 ) );
    assert( (M1 + offx) <= M2 );
    assert( (N1 + offy) <= N2 );

    ldau = (A->rk == -1) ? A->rkmax : M1;
    ldav = (transAv == PastixNoTrans) ? A->rkmax : N1;

    core_zlrsze( 0, M2, N2, B, A->rk, -1, -1 );
    u = B->u;
    v = B->v;

    B->rk = A->rk;

    if ( A->rk == -1 ) {
        if ( (M1 != M2) || (N1 != N2) ) {
            LAPACKE_zlaset_work( LAPACK_COL_MAJOR, 'A', M2, N2,
                                 0.0, 0.0, u, M2 );
        }
        ret = core_zgeadd( PastixNoTrans, M1, N1,
                           alpha, A->u, ldau,
                           0.0, u + M2 * offy + offx, M2 );
        assert(ret == 0);
    }
    else {
        if ( M1 != M2 ) {
            LAPACKE_zlaset_work( LAPACK_COL_MAJOR, 'A', M2, B->rk,
                                 0.0, 0.0, u, M2 );
        }
        ret = LAPACKE_zlacpy_work( LAPACK_COL_MAJOR, 'A', M1, A->rk,
                                   A->u, ldau,
                                   u + offx, M2 );
        assert(ret == 0);

        if ( N1 != N2 ) {
            LAPACKE_zlaset_work( LAPACK_COL_MAJOR, 'A', B->rk, N2,
                                 0.0, 0.0, v, B->rkmax );
        }
        ret = core_zgeadd( transAv, A->rk, N1,
                           alpha, A->v, ldav,
                           0.0, v + B->rkmax * offy, B->rkmax );
        assert(ret == 0);
    }

#if 0
    {
        pastix_complex64_t *work = malloc( M2 * N2 * sizeof(pastix_complex64_t) );

        core_zlr2ge( PastixNoTrans, M2, N2, B, work, M2 );

        lowrank->core_ge2lr( lowrank->tolerance, -1, M2, N2, work, M2, B );

        free(work);
    }
#endif

    (void) lowrank;
    (void) ret;
}


/**
 *******************************************************************************
 *
 * @brief Concatenate left parts of two low-rank matrices
 *
 *******************************************************************************
 *
 * @param[in] alpha
 *          alpha * A is add to B
 *
 * @param[in] M1
 *          The number of rows of the matrix A.
 *
 * @param[in] N1
 *          The number of columns of the matrix A.
 *
 * @param[in] A
 *          The low-rank representation of the matrix A.
 *
 * @param[in] M2
 *          The number of rows of the matrix B.
 *
 * @param[in] B
 *          The low-rank representation of the matrix B.
 *
 * @param[in] offx
 *          The horizontal offset of A with respect to B.
 *
 * @param[inout] u1u2
 *          The workspace where matrices are concatenated
 *
 *******************************************************************************/
void
core_zlrconcatenate_u( pastix_complex64_t alpha,
                       pastix_int_t M1, pastix_int_t N1, const pastix_lrblock_t *A,
                       pastix_int_t M2,                        pastix_lrblock_t *B,
                       pastix_int_t offx,
                       pastix_complex64_t *u1u2 )
{
    pastix_complex64_t *tmp;
    pastix_int_t i, ret, rank;
    pastix_int_t ldau, ldbu;

    rank = (A->rk == -1) ? pastix_imin(M1, N1) : A->rk;
    rank += B->rk;

    ldau = (A->rk == -1) ? A->rkmax : M1;
    ldbu = M2;

    ret = LAPACKE_zlacpy_work( LAPACK_COL_MAJOR, 'A', M2, B->rk,
                               B->u, ldbu, u1u2, M2 );
    assert(ret == 0);

    tmp = u1u2 + B->rk * M2;
    if ( A->rk == -1 ) {
        /*
         * A is full of rank M1, so A will be integrated into v1v2
         */
        if ( M1 < N1 ) {
            if ( M1 != M2 ) {
                /* Set to 0 */
                memset(tmp, 0, M2 * M1 * sizeof(pastix_complex64_t));

                /* Set diagonal */
                tmp += offx;
                for (i=0; i<M1; i++, tmp += M2+1) {
                    *tmp = 1.0;
                }
            }
            else {
                assert( offx == 0 );
                ret = LAPACKE_zlaset_work( LAPACK_COL_MAJOR, 'A', M2, M1,
                                           0.0, 1.0, tmp, M2 );
                assert( ret == 0 );
            }
        }
        else {
            /*
             * A is full of rank N1, so A is integrated into u1u2
             */
            if ( M1 != M2 ) {
                memset(tmp, 0, M2 * N1 * sizeof(pastix_complex64_t));
            }
            ret = LAPACKE_zlacpy_work( LAPACK_COL_MAJOR, 'A', M1, N1,
                                       A->u, ldau, tmp + offx, M2 );
            assert(ret == 0);
        }
    }
    /*
     * A is low rank of rank A->rk
     */
    else {
        if ( M1 != M2 ) {
            memset(tmp, 0, M2 * A->rk * sizeof(pastix_complex64_t));
        }
        ret = LAPACKE_zlacpy_work( LAPACK_COL_MAJOR, 'A', M1, A->rk,
                                   A->u, ldau, tmp + offx, M2 );
        assert(ret == 0);
    }
    (void) ret;
    (void) alpha;
}


/**
 *******************************************************************************
 *
 * @brief Concatenate right parts of two low-rank matrices
 *
 *******************************************************************************
 *
 * @param[in] transA1
 *         @arg PastixNoTrans:  No transpose, op( A ) = A;
 *         @arg PastixTrans:  Transpose, op( A ) = A';
 *
 * @param[in] alpha
 *          alpha * A is add to B
 *
 * @param[in] M1
 *          The number of rows of the matrix A.
 *
 * @param[in] N1
 *          The number of columns of the matrix A.
 *
 * @param[in] A
 *          The low-rank representation of the matrix A.
 *
 * @param[in] N2
 *          The number of columns of the matrix B.
 *
 * @param[in] B
 *          The low-rank representation of the matrix B.
 *
 * @param[in] offy
 *          The vertical offset of A with respect to B.
 *
 * @param[inout] v1v2
 *          The workspace where matrices are concatenated
 *
 *******************************************************************************/
void
core_zlrconcatenate_v( pastix_trans_t transA1, pastix_complex64_t alpha,
                       pastix_int_t M1, pastix_int_t N1, const pastix_lrblock_t *A,
                                        pastix_int_t N2,       pastix_lrblock_t *B,
                       pastix_int_t offy,
                       pastix_complex64_t *v1v2 )
{
    pastix_complex64_t *tmp;
    pastix_int_t i, ret, rank;
    pastix_int_t ldau, ldav, ldbv;

    rank = (A->rk == -1) ? pastix_imin(M1, N1) : A->rk;
    rank += B->rk;

    ldau = (A->rk == -1) ? A->rkmax : M1;
    ldav = (transA1 == PastixNoTrans) ? A->rkmax : N1;
    ldbv = B->rkmax;

    ret = LAPACKE_zlacpy_work( LAPACK_COL_MAJOR, 'A', B->rk, N2,
                               B->v, ldbv, v1v2, rank );
    assert(ret == 0);

    tmp = v1v2 + B->rk;
    if ( A->rk == -1 ) {
        assert( transA1 == PastixNoTrans );
        /*
         * A is full of rank M1, so it is integrated into v1v2
         */
        if ( M1 < N1 ) {
            if (N1 != N2) {
                ret = LAPACKE_zlaset_work( LAPACK_COL_MAJOR, 'A', M1, N2,
                                           0.0, 0.0, tmp, rank );
                assert( ret == 0 );
            }
            core_zgeadd( PastixNoTrans, M1, N1,
                         alpha, A->u, ldau,
                         0.0, tmp + offy * rank, rank );
        }
        /*
         * A is full of rank N1, so it has been integrated into u1u2
         */
        else {
            if (N1 != N2) {
                /* Set to 0 */
                ret = LAPACKE_zlaset_work( LAPACK_COL_MAJOR, 'A', N1, N2,
                                           0.0, 0.0, tmp, rank );
                assert(ret == 0);

                /* Set diagonal */
                tmp += offy * rank;
                for (i=0; i<N1; i++, tmp += rank+1) {
                    *tmp = alpha;
                }
            }
            else {
                assert( offy == 0 );
                ret = LAPACKE_zlaset_work( LAPACK_COL_MAJOR, 'A', N1, N2,
                                           0.0, alpha, tmp + offy * rank, rank );
                assert( ret == 0 );
            }
        }
    }
    /*
     * A is low rank of rank A->rk
     */
    else {
        if (N1 != N2) {
            ret = LAPACKE_zlaset_work( LAPACK_COL_MAJOR, 'A', A->rk, N2,
                                       0.0, 0.0, tmp, rank );
            assert(ret == 0);
        }
        core_zgeadd( transA1, A->rk, N1,
                     alpha, A->v,              ldav,
                       0.0, tmp + offy * rank, rank );
    }
    (void) ret;
    (void) alpha;
}

/**
 *******************************************************************************
 *
 * @brief Template to convert a full rank matrix into a low rank matrix through
 * QR decompositions
 *
 *******************************************************************************
 *
 * @param[in] rrqrfct
 *          QR decomposition function used to compute the rank revealing
 *          factorization and create the low-rank form of A.
 *
 * @param[in] tol
 *          The tolerance used as a criterai to eliminate information from the
 *          full rank matrix
 *
 * @param[in] rklimit
 *          The maximum rank to store the matrix in low-rank format. If
 *          -1, set to min(m, n) / PASTIX_LR_MINRATIO.
 *
 * @param[in] m
 *          Number of rows of the matrix A, and of the low rank matrix Alr.
 *
 * @param[in] n
 *          Number of columns of the matrix A, and of the low rank matrix Alr.
 *
 * @param[in] A
 *          The matrix of dimension lda-by-n that needs to be compressed
 *
 * @param[in] lda
 *          The leading dimension of the matrix A. lda >= max(1, m)
 *
 * @param[out] Alr
 *          The low rank matrix structure that will store the low rank
 *          representation of A
 *
 *******************************************************************************/
pastix_fixdbl_t
core_zge2lr_qr( core_zrrqr_t rrqrfct,
                pastix_fixdbl_t tol, pastix_int_t rklimit,
                pastix_int_t m, pastix_int_t n,
                const void *Avoid, pastix_int_t lda,
                pastix_lrblock_t *Alr )
{
    int                 ret;
    pastix_int_t        nb = 32;
    pastix_complex64_t *A = (pastix_complex64_t*)Avoid;
    pastix_int_t        lwork;
    pastix_complex64_t *work, *tau, zzsize;
    double             *rwork;
    pastix_int_t       *jpvt;
    pastix_int_t        zsize, rsize;
    pastix_complex64_t *zwork;
    pastix_fixdbl_t     flops;

    double norm = LAPACKE_zlange_work( LAPACK_COL_MAJOR, 'f', m, n,
                                       A, lda, NULL );

    /* work */
    rklimit = ( rklimit < 0 ) ? core_get_rklimit( m, n ) : rklimit;
    ret = rrqrfct( tol * norm, rklimit, 1, nb,
                   m, n, NULL, m,
                   NULL, NULL,
                   &zzsize, -1, NULL );

    lwork = (pastix_int_t)zzsize;
    zsize = lwork;
    /* tau */
    zsize += n;

    /* rwork */
    rsize = 2 * n;

#if defined(PASTIX_DEBUG_LR)
    zwork = NULL;

    tau   = malloc( n     * sizeof(pastix_complex64_t) );
    work  = malloc( lwork * sizeof(pastix_complex64_t) );
    rwork = malloc( rsize * sizeof(double) );
#else
    zwork = malloc( zsize * sizeof(pastix_complex64_t) + rsize * sizeof(double) );

    tau   = zwork;
    work  = tau + n;
    rwork = (double*)(work + lwork);
#endif

    jpvt = malloc( n * sizeof(pastix_int_t) );

    /**
     * Allocate the Low rank matrix in full-rank to store the copy of A
     */
    core_zlralloc( m, n, -1, Alr );

    ret = LAPACKE_zlacpy_work( LAPACK_COL_MAJOR, 'A', m, n,
                               A, lda, Alr->u, m );
    assert(ret == 0);

    ret = rrqrfct( tol * norm, rklimit, 1, nb,
                   m, n, Alr->u, m,
                   jpvt, tau,
                   work, lwork, rwork );
    if (ret == -1) {
        flops = FLOPS_ZGEQRF( m, n );
    }
    else {
        flops = FLOPS_ZGEQRF( m, ret ) + FLOPS_ZUNMQR( m, n-ret, ret, PastixLeft );
    }

    /**
     * It was not interesting to compress, so we restore the dense version in Alr
     */
    if ( ret == -1 ) {
        ret = LAPACKE_zlacpy_work( LAPACK_COL_MAJOR, 'A', m, n,
                                   A, lda, Alr->u, Alr->rkmax );
        assert(ret == 0);
    }
    else if ( ret == 0 ) {
        core_zlrsze( 0, m, n, Alr, 0, 0, rklimit );
    }
    /**
     * We compute U and V
     */
    else {
        pastix_int_t i;
        pastix_complex64_t *Acpy = Alr->u;
        pastix_complex64_t *U, *V;

        assert( ret > 0 );

        /* Resize Alr */
        Alr->u = NULL;
        core_zlrsze( 0, m, n, Alr, ret, ret, rklimit );
        U = Alr->u;
        V = Alr->v;

        /* Compute the final U form */
        ret = LAPACKE_zlacpy_work( LAPACK_COL_MAJOR, 'A', m, Alr->rk,
                                   Acpy, m, U, m );
        assert(ret == 0);

        ret = LAPACKE_zungqr_work( LAPACK_COL_MAJOR, m, Alr->rk, Alr->rk,
                                   U, m, tau, work, lwork );
        assert(ret == 0);
        flops += FLOPS_ZUNGQR( m, Alr->rk, Alr->rk );

        /* Compute the final V form */
        ret = LAPACKE_zlaset_work( LAPACK_COL_MAJOR, 'L', Alr->rk-1, Alr->rk-1,
                                   0.0, 0.0, Acpy + 1, m );

        for (i=0; i<n; i++){
            memcpy( V    + jpvt[i] * Alr->rk,
                    Acpy + i       * m,
                    Alr->rk * sizeof(pastix_complex64_t) );
        }

        free( Acpy );
    }

#if defined(PASTIX_DEBUG_LR)
    if ( Alr->rk > 0 ) {
        int rc = core_zlrdbg_check_orthogonality( m, Alr->rk, Alr->u, m );
        if (rc == 1) {
            fprintf(stderr, "Failed to compress a matrix and generate an orthogonal u\n" );
        }
    }
#endif

    free( zwork );
    free( jpvt );
#if defined(PASTIX_DEBUG_LR)
    free( tau   );
    free( work  );
    free( rwork );
#endif
    return flops;
}

/**
 *******************************************************************************
 *
 * @brief Template to perform the addition of two low-rank structures with
 * compression kernel based on QR decomposition.
 *
 * Add two LR structures A=(-u1) v1^T and B=u2 v2^T into u2 v2^T
 *
 *    u2v2^T - u1v1^T = (u2 u1) (v2 v1)^T
 *    Orthogonalize (u2 u1) = (u2, u1 - u2(u2^T u1)) * (I u2^T u1)
 *                                                     (0    I   )
 *    Compute Rank Revealing QR decomposition of (I u2^T u1) * (v2 v1)^T
 *                                               (0    I   )
 * Any QR rank revealing kernel can be used for the recompression of the V part.
 *
 *******************************************************************************
 *
 * @param[in] rrqrfct
 *          QR decomposition function used to compute the rank revealing
 *          factorization of the sum of the two low-rank matrices.
 *
 * @param[in] lowrank
 *          The structure with low-rank parameters.
 *
 * @param[in] transA1
 *         @arg PastixNoTrans:  No transpose, op( A ) = A;
 *         @arg PastixTrans:  Transpose, op( A ) = A';
 *
 * @param[in] alpha
 *          alpha * A is add to B
 *
 * @param[in] M1
 *          The number of rows of the matrix A.
 *
 * @param[in] N1
 *          The number of columns of the matrix A.
 *
 * @param[in] A
 *          The low-rank representation of the matrix A.
 *
 * @param[in] M2
 *          The number of rows of the matrix B.
 *
 * @param[in] N2
 *          The number of columns of the matrix B.
 *
 * @param[in] B
 *          The low-rank representation of the matrix B.
 *
 * @param[in] offx
 *          The horizontal offset of A with respect to B.
 *
 * @param[in] offy
 *          The vertical offset of A with respect to B.
 *
 *******************************************************************************
 *
 * @return  The new rank of u2 v2^T or -1 if ranks are too large for
 *          recompression
 *
 *******************************************************************************/
pastix_fixdbl_t
core_zrradd_qr( core_zrrqr_t rrqrfct,
                const pastix_lr_t *lowrank, pastix_trans_t transA1, const void *alphaptr,
                pastix_int_t M1, pastix_int_t N1, const pastix_lrblock_t *A,
                pastix_int_t M2, pastix_int_t N2,       pastix_lrblock_t *B,
                pastix_int_t offx, pastix_int_t offy )
{
    pastix_int_t rankA, rank, M, N, minV;
    pastix_int_t i, ret, new_rank, rklimit;
    pastix_int_t ldau, ldav, ldbu, ldbv, ldu, ldv;
    pastix_complex64_t *u1u2, *v1v2, *u;
    pastix_complex64_t *zbuf, *tauV;
    size_t wzsize;
    double norm;
    double tol = lowrank->tolerance;

    /* PQRCP parameters / workspace */
    pastix_int_t        nb = 32;
    pastix_int_t        lwork;
    pastix_int_t       *jpvt;
    pastix_complex64_t *zwork, zzsize;
    double             *rwork;
    pastix_complex64_t  alpha = *((pastix_complex64_t*)alphaptr);
    pastix_fixdbl_t     flops, total_flops = 0.;

#if defined(PASTIX_DEBUG_LR)
    if ( B->rk > 0 ) {
        int rc = core_zlrdbg_check_orthogonality( M2, B->rk, B->u, M2 );
        if (rc == 1) {
            fprintf(stderr, "Failed to have B->u orthogonal in entry of rradd\n" );
        }
    }
#endif

    rankA = (A->rk == -1) ? pastix_imin(M1, N1) : A->rk;
    rank  = rankA + B->rk;
    M = pastix_imax(M2, M1);
    N = pastix_imax(N2, N1);

    minV = pastix_imin(N, rank);

    assert(M2 == M && N2 == N);
    assert(B->rk != -1);

    assert( A->rk <= A->rkmax);
    assert( B->rk <= B->rkmax);

    if ( ((M1 + offx) > M2) ||
         ((N1 + offy) > N2) )
    {
        errorPrint("Dimensions are not correct");
        assert(0 /* Incorrect dimensions */);
        return total_flops;
    }

    /*
     * A is rank null, nothing to do
     */
    if ( A->rk == 0 ) {
        return total_flops;
    }

    /*
     * Let's handle case where B is a null matrix
     *   B = alpha A
     */
    if ( B->rk == 0 ) {
        core_zlrcpy( lowrank, transA1, alpha,
                     M1, N1, A, M2, N2, B,
                     offx, offy );
        return total_flops;
    }

    /*
     * The rank is too big, let's try to compress
     */
    if ( rank > pastix_imin( M, N ) ) {
        assert(0);
    }

    /*
     * Let's define leading dimensions
     */
    ldau = (A->rk == -1) ? A->rkmax : M1;
    ldav = (transA1 == PastixNoTrans) ? A->rkmax : N1;
    ldbu = M;
    ldbv = B->rkmax;
    ldu = M;
    ldv = rank;

    /*
     * Let's compute the size of the workspace
     */
    /* u1u2 and v1v2 */
    wzsize = (M+N) * rank;
    /* tauV */
    wzsize += minV;

    /* RRQR workspaces */
    rklimit = pastix_imin( rank, core_get_rklimit( M, N ) );
    rrqrfct( tol, rklimit, 1, nb,
             rank, N, NULL, ldv,
             NULL, NULL,
             &zzsize, -1, NULL );
    lwork = (pastix_int_t)(zzsize);
    wzsize += lwork;

#if defined(PASTIX_DEBUG_LR)
    zbuf = NULL;
    u1u2  = malloc( ldu * rank * sizeof(pastix_complex64_t) );
    v1v2  = malloc( ldv * N    * sizeof(pastix_complex64_t) );
    tauV  = malloc( rank       * sizeof(pastix_complex64_t) );
    zwork = malloc( lwork      * sizeof(pastix_complex64_t) );

    rwork = malloc( 2 * pastix_imax( rank, N ) * sizeof(double) );
#else
    zbuf = malloc( wzsize * sizeof(pastix_complex64_t) + 2 * pastix_imax(rank, N) * sizeof(double) );

    u1u2  = zbuf;
    v1v2  = u1u2 + ldu * rank;
    tauV  = v1v2 + ldv * N;
    zwork = tauV + rank;

    rwork = (double*)(zwork + lwork);
#endif

    /*
     * Concatenate U2 and U1 in u1u2
     *  [ u2  0  ]
     *  [ u2  u1 ]
     *  [ u2  0  ]
     */
    core_zlrconcatenate_u( alpha,
                           M1, N1, A,
                           M2,     B,
                           offx, u1u2 );

    /*
     * Concatenate V2 and V1 in v1v2
     *  [ v2^h v2^h v2^h ]
     *  [ 0    v1^h 0    ]
     */
    core_zlrconcatenate_v( transA1, alpha,
                           M1, N1, A,
                               N2, B,
                           offy, v1v2 );

    /*
     * Orthogonalize [u2, u1]
     * [u2, u1] = [u2, u1 - u2(u2Tu1)] * (I u2Tu1)
     *                                   (0   I  )
     */

    /* We do not care is A was integrated into v1v2 */
    if (rankA != 0) {

        kernel_trace_start_lvl2( PastixKernelLvl2_LR_add2C_rradd_orthogonalize );
        switch ( pastix_lr_ortho ) {
        case PastixCompressOrthoQR:
            flops = core_zlrorthu_fullqr( M, N, B->rk + rankA,
                                          u1u2, ldu, v1v2, ldv );
            break;

        case PastixCompressOrthoPartialQR:
            flops = core_zlrorthu_partialqr( M, N, B->rk, &rankA, offx, offy,
                                             u1u2, ldu, v1v2, ldv );
            break;

        case PastixCompressOrthoCGS:
            pastix_attr_fallthrough;

        default:
            flops = core_zlrorthu_cgs( M2, N2, M1, N1, B->rk, &rankA, offx, offy,
                                       u1u2, ldu, v1v2, ldv );
        }
        kernel_trace_stop_lvl2( flops );

        total_flops += flops;
    }

    rank = B->rk + rankA;

    if (rankA == 0) {
        /*
         * The final B += A fit in B
         * Lets copy and return
         */
        LAPACKE_zlacpy_work( LAPACK_COL_MAJOR, 'A', M, B->rk, u1u2, ldu, B->u, ldbu );
        LAPACKE_zlacpy_work( LAPACK_COL_MAJOR, 'A', B->rk, N, v1v2, ldv, B->v, ldbv );

        free(zbuf);
#if defined(PASTIX_DEBUG_LR)
        free( u1u2  );
        free( v1v2  );
        free( tauV  );
        free( zwork );
        free( rwork );
#endif
        return total_flops;
    }

    MALLOC_INTERN( jpvt, pastix_imax(rank, N), pastix_int_t );

    norm = LAPACKE_zlange_work( LAPACK_COL_MAJOR, 'f', rank, N,
                                v1v2, ldv, NULL );

    /*
     * Perform RRQR factorization on (I u2Tu1) v1v2 = (Q2 R2)
     *                               (0   I  )
     */
    kernel_trace_start_lvl2( PastixKernelLvl2_LR_add2C_rradd_recompression );
    rklimit = pastix_imin( rklimit, rank );
    new_rank = rrqrfct( tol * norm, rklimit, 1, nb,
                        rank, N, v1v2, ldv,
                        jpvt, tauV,
                        zwork, lwork, rwork );
    flops = (new_rank == -1) ? FLOPS_ZGEQRF( rank, N )
        :                     (FLOPS_ZGEQRF( rank, new_rank ) +
                               FLOPS_ZUNMQR( rank, N-new_rank, new_rank, PastixLeft ));
    kernel_trace_stop_lvl2_rank( flops, new_rank );
    total_flops += flops;

    /*
     * First case: The rank is too big, so we decide to uncompress the result
     */
    if ( (new_rank > rklimit) ||
         (new_rank == -1) )
    {
        pastix_lrblock_t Bbackup = *B;

        core_zlralloc( M, N, -1, B );
        u = B->u;

        /* Uncompress B */
        flops = FLOPS_ZGEMM( M, N, Bbackup.rk );
        kernel_trace_start_lvl2( PastixKernelLvl2_LR_add2C_uncompress );
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    M, N, Bbackup.rk,
                    CBLAS_SADDR(zone),  Bbackup.u, ldbu,
                                        Bbackup.v, ldbv,
                    CBLAS_SADDR(zzero), u, M );
        kernel_trace_stop_lvl2( flops );
        total_flops += flops;

        /* Add A into it */
        if ( A->rk == -1 ) {
            flops = 2 * M1 * N1;
            kernel_trace_start_lvl2( PastixKernelLvl2_FR_GEMM );
            core_zgeadd( transA1, M1, N1,
                         alpha, A->u, ldau,
                         zone, u + offy * M + offx, M);
            kernel_trace_stop_lvl2( flops );
        }
        else {
            flops = FLOPS_ZGEMM( M1, N1, A->rk );
            kernel_trace_start_lvl2( PastixKernelLvl2_FR_GEMM );
            cblas_zgemm(CblasColMajor, CblasNoTrans, (CBLAS_TRANSPOSE)transA1,
                        M1, N1, A->rk,
                        CBLAS_SADDR(alpha), A->u, ldau,
                                            A->v, ldav,
                        CBLAS_SADDR(zone), u + offy * M + offx, M);
            kernel_trace_stop_lvl2( flops );
        }
        total_flops += flops;
        core_zlrfree(&Bbackup);
        free( zbuf );
        free( jpvt );
#if defined(PASTIX_DEBUG_LR)
        free( u1u2  );
        free( v1v2  );
        free( tauV  );
        free( zwork );
        free( rwork );
#endif
        return total_flops;
    }
    else if ( new_rank == 0 ) {
        core_zlrfree(B);
        free( zbuf );
        free( jpvt );
#if defined(PASTIX_DEBUG_LR)
        free( u1u2  );
        free( v1v2  );
        free( tauV  );
        free( zwork );
        free( rwork );
#endif
        return total_flops;
    }

    /*
     * We need to reallocate the buffer to store the new compressed version of B
     * because it wasn't big enough
     */
    ret = core_zlrsze( 0, M, N, B, new_rank, -1, -1 );
    assert( ret != -1 );
    assert( B->rkmax >= new_rank );
    assert( B->rkmax >= B->rk    );

    ldbv = B->rkmax;

    /* B->v = P v1v2 */
    {
        pastix_complex64_t *tmpV;
        pastix_int_t lm;

        memset(B->v, 0, N * ldbv * sizeof(pastix_complex64_t));
        tmpV = B->v;
        for (i=0; i<N; i++){
            lm = pastix_imin( new_rank, i+1 );
            memcpy(tmpV + jpvt[i] * ldbv,
                   v1v2 + i       * ldv,
                   lm * sizeof(pastix_complex64_t));
        }
    }

    /* Compute Q2 factor */
    {
        flops = FLOPS_ZUNGQR( rank, new_rank, new_rank )
            +   FLOPS_ZGEMM( M, new_rank, rank );

        kernel_trace_start_lvl2( PastixKernelLvl2_LR_add2C_rradd_computeNewU );
        ret = LAPACKE_zungqr_work( LAPACK_COL_MAJOR, rank, new_rank, new_rank,
                                   v1v2, ldv, tauV, zwork, lwork );
        assert(ret == 0);

        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    M, new_rank, rank,
                    CBLAS_SADDR(zone),  u1u2, ldu,
                                        v1v2, ldv,
                    CBLAS_SADDR(zzero), B->u, ldbu);
        kernel_trace_stop_lvl2( flops );
        total_flops += flops;
    }

    free( zbuf );
    free( jpvt );

#if defined(PASTIX_DEBUG_LR)
    free( u1u2  );
    free( v1v2  );
    free( tauV  );
    free( zwork );
    free( rwork );
#endif

    (void)ret;
    return total_flops;
}
