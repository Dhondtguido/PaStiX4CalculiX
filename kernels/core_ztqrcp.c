/**
 *
 * @file core_ztqrcp.c
 *
 * PaStiX implementation of the truncated rank-revealing QR with column pivoting
 * based on Lapack GEQP3.
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
#include "blend/solver.h"
#include "pastix_zcores.h"
#include "pastix_zlrcores.h"
#include "z_nan_check.h"

#ifndef DOXYGEN_SHOULD_SKIP_THIS
static pastix_complex64_t mzone = -1.0;
static pastix_complex64_t zone  =  1.0;
static pastix_complex64_t zzero =  0.0;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/**
 *******************************************************************************
 *
 * @brief Compute a randomized QR factorization with truncated updates.
 *
 * This routine is originated from ???.
 *
 *******************************************************************************
 *
 * @param[in] tol
 *          The relative tolerance criterion. Computations are stopped when the
 *          frobenius norm of the residual matrix is lower than tol.
 *          If tol < 0, then maxrank reflectors are computed.
 *
 * @param[in] maxrank
 *          Maximum number of reflectors computed. Computations are stopped when
 *          the rank exceeds maxrank. If maxrank < 0, all reflectors are computed
 *          or up to the tolerance criterion.
 *
 * @param[in] unused
 *          Unused parameter in this kernel added to match API of RQRCP and PQRCP.
 *
 * @param[in] nb
 *          Tuning parameter for the GEMM blocking size. if nb < 0, nb is set to
 *          32.
 *
 * @param[in] m
 *          Number of rows of the matrix A.
 *
 * @param[in] n
 *          Number of columns of the matrix A.
 *
 * @param[in] A
 *          The matrix of dimension lda-by-n that needs to be compressed.
 *
 * @param[in] lda
 *          The leading dimension of the matrix A. lda >= max(1, m).
 *
 * @param[out] jpvt
 *          The array that describes the permutation of A.
 *
 * @param[out] tau
 *          Contains scalar factors of the elementary reflectors for the matrix
 *          Q.
 *
 * @param[in] work
 *          Workspace array of size lwork.
 *
 * @param[in] lwork
 *          The dimension of the work area. lwork >= (nb * n + max(n, m) )
 *          If lwork == -1, the functions returns immediately and work[0]
 *          contains the optimal size of work.
 *
 * @param[in] rwork
 *          Workspace array used to store partial and exact column norms (2-by-n)
 *
 *******************************************************************************
 *
 * @return This routine will return the rank of A (>=0) or -1 if it didn't
 *         manage to compress within the margins of tolerance and maximum rank.
 *
 *******************************************************************************/
int
core_ztqrcp( double tol, pastix_int_t maxrank, int refine, pastix_int_t nb,
             pastix_int_t m, pastix_int_t n,
             pastix_complex64_t *A, pastix_int_t lda,
             pastix_int_t *jpvt, pastix_complex64_t *tau,
             pastix_complex64_t *work, pastix_int_t lwork,  double *rwork )
{
    int                 SEED[4] = {26, 67, 52, 197};
    pastix_int_t        j, k, in, itmp, d, ib, loop = 1;
    int                 ret;
    pastix_int_t        minMN, lwkopt;
    pastix_int_t        b = 24;
    pastix_int_t        p = 8;
    pastix_int_t        bp = b + p;
    pastix_int_t        size_B, size_O, size_W, size_Y, size_A, size_T, sublw;
    pastix_int_t        ldb, ldw, ldy;
    pastix_int_t       *jpvt_b;
    pastix_int_t        rk;
    double              tolB = sqrt( (double)(bp) ) * tol;
    pastix_complex64_t *AP, *Y, *WT, *T, *B, *tau_b, *omega, *subw;

    if ( nb < 0 ) {
        nb = 32;
    }

    minMN = pastix_imin(m, n);
    if ( maxrank < 0 ) {
        maxrank = minMN;
    }
    maxrank = pastix_imin( maxrank, minMN );

    ldb = bp;
    ldw = maxrank;

    size_B = ldb * n;
    size_O = ldb * m;
    size_W = n * maxrank;
    size_Y = b * b;
    ldy = b;
    size_A = m * n;
    size_T = b * b;

    sublw = n * nb + pastix_imax( bp, n );     /* pqrcp */
    sublw = pastix_imax( sublw, size_O );      /* Omega */
    sublw = pastix_imax( sublw, b * maxrank ); /* update */

    lwkopt = size_A + size_Y + size_W
        +    size_T + size_B + n + sublw;

    if ( lwork == -1 ) {
        work[0] = (pastix_complex64_t)lwkopt;
        return 0;
    }
#if !defined(NDEBUG)
    if (m < 0) {
        return -1;
    }
    if (n < 0) {
        return -2;
    }
    if (lda < pastix_imax(1, m)) {
        return -4;
    }
    if( lwork < lwkopt ) {
        return -8;
    }
#endif

    if ( (minMN == 0) || (maxrank == 0) ) {
        return 0;
    }

    jpvt_b = malloc( n * sizeof(pastix_int_t) );

    AP    = work;
    Y     = AP + size_A;
    WT    = Y  + size_Y;
    T     = WT + size_W;
    B     = T  + size_T;
    tau_b = B  + size_B;
    omega = tau_b + n;
    subw  = tau_b + n;

    /* Initialize diagonal block of Housholders reflectors */
    ret = LAPACKE_zlaset_work( LAPACK_COL_MAJOR, 'A', b, b,
                               0., 1., Y, ldy );
    assert( ret == 0 );

    /* Initialize T */
    memset(T, 0, size_T * sizeof(pastix_complex64_t));

    /* Backup A */
    ret = LAPACKE_zlacpy_work( LAPACK_COL_MAJOR, 'A', m, n,
                               A, lda, AP, m );
    assert( ret == 0 );

    /* Initialize pivots */
    for (j=0; j<n; j++) jpvt[j] = j;

    /*
     * Computation of the Gaussian matrix
     */
    LAPACKE_zlarnv_work(3, SEED, size_O, omega);
    cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                 bp, n, m,
                 CBLAS_SADDR(zone),  omega, bp,
                                     A,     lda,
                 CBLAS_SADDR(zzero), B,     ldb );

    rk = 0;
    d  = 0;
    while ( (rk < maxrank) && loop )
    {
        ib = pastix_imin( b, maxrank-rk );
        d = core_zpqrcp( tolB, ib, 1, nb,
                         bp, n-rk,
                         B      + rk * ldb, ldb,
                         jpvt_b + rk, tau_b,
                         subw, sublw, rwork );

        /* If fails to reach the tolerance before maxrank, let's restore the max value */
        if ( d == -1 ) {
            d = ib;
        }
        if ( d < ib ) {
            loop = 0;
        }
        if ( d == 0 ) {
            break;
        }

        /* Updating jpvt, A, and AP */
        for (j = rk; j < rk + d; j++) {
            if (jpvt_b[j] >= 0) {
                k = j;
                in = jpvt_b[k] + rk;

                /* Mark as done */
                jpvt_b[k] = - jpvt_b[k] - 1;

                while( jpvt_b[in] >= 0 ) {

                    if (k != in) {
                        cblas_zswap( m, A  + k  * lda, 1,
                                        A  + in * lda, 1 );
                        cblas_zswap( m, AP + k  * m,   1,
                                        AP + in * m,   1 );

                        itmp     = jpvt[k];
                        jpvt[k]  = jpvt[in];
                        jpvt[in] = itmp;

                        if (rk > 0) {
                            cblas_zswap( rk, WT + k  * ldw, 1,
                                             WT + in * ldw, 1 );
                        }
                    }
                    itmp = jpvt_b[in];
                    jpvt_b[in] = - jpvt_b[in] - 1;
                    k = in;
                    in = itmp + rk;
                }
            }
        }

        if (rk > 0) {
            /* Update the selected columns before factorization */
            cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                         m-rk, d, rk,
                         CBLAS_SADDR(mzone), A  + rk,            lda,
                                             WT + rk * ldw,      ldw,
                         CBLAS_SADDR(zone),  A  + rk * lda + rk, lda );
        }

        /*
         * Factorize the d selected columns of A without pivoting
         */
        ret = LAPACKE_zgeqrf_work( LAPACK_COL_MAJOR, m-rk, d,
                                   A + rk * lda + rk, lda, tau + rk,
                                   work, lwork );
        assert( ret == 0 );

        ret = LAPACKE_zlarft( LAPACK_COL_MAJOR, 'F', 'C', m-rk, d,
                              A + rk * lda + rk, lda, tau + rk, T, b );
        assert( ret == 0 );

        /*
         * Compute the update line 11 of algorithm 6 in "Randomized QR with
         * Column pivoting" from Duersch and Gu
         *
         * W_2^h = T^h ( Y_2^h * A - (Y_2^h * Y) * W_1^h )
         *
         * Step 1: Y_2^h * A
         *     a) W[rk:rk+d] <- A
         *     b) W[rk:rk+d] <- Y_2^h * A, split in triangular part + rectangular part
         */
        ret = LAPACKE_zlacpy_work( LAPACK_COL_MAJOR, 'L', d-1, d-1,
                                   A + lda * rk + rk + 1, lda,
                                   Y +                 1, ldy );
        assert( ret == 0 );

        /* Triangular part */
        cblas_zgemm( CblasColMajor, CblasConjTrans, CblasNoTrans,
                     d, n, d,
                     CBLAS_SADDR(zone),  Y,       ldy,
                                         AP + rk, m,
                     CBLAS_SADDR(zzero), WT + rk, ldw );

        /* Rectangular part */
        if ( rk + d < m ) {
            cblas_zgemm( CblasColMajor, CblasConjTrans, CblasNoTrans,
                         d, n, m-rk-d,
                         CBLAS_SADDR(zone), A  + rk * lda + rk + d, lda,
                                            AP +            rk + d, m,
                         CBLAS_SADDR(zone), WT +            rk,     ldw );
        }

        /*
         * Step 2: (Y_2^h * A) - (Y_2^h * Y) * W_1^h
         *     a) work = (Y_2^h * Y)
         *     b) (Y_2^h * A) - work * W_1^h
         */
        if ( rk > 0 ) {
            /* Triangular part */
            cblas_zgemm( CblasColMajor, CblasConjTrans, CblasNoTrans,
                         d, rk, d,
                         CBLAS_SADDR(zone),  Y,      ldy,
                                             A + rk, lda,
                         CBLAS_SADDR(zzero), subw,   d );

            /* Rectangular part */
            if ( rk + d < m ) {
                cblas_zgemm( CblasColMajor, CblasConjTrans, CblasNoTrans,
                             d, rk, m-rk-d,
                             CBLAS_SADDR(zone), A  + rk * lda + rk + d, lda,
                                                A  +            rk + d, lda,
                             CBLAS_SADDR(zone), subw,                   d );
            }

            cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                         d, n, rk,
                         CBLAS_SADDR(mzone), subw,    d,
                                             WT,      ldw,
                         CBLAS_SADDR(zone),  WT + rk, ldw );
        }

        /*
         * Step 3: W_2^h = T^h ( Y_2^h * A - (Y_2^h * Y) * W_1^h )
         *      W_2^h = T^h W_2^h
         */
        cblas_ztrmm( CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, CblasNonUnit,
                     d, n, CBLAS_SADDR(zone),
                     T,       b,
                     WT + rk, ldw );

        /* Update current d rows of R */
        if ( rk+d < n ) {
            cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                         d, n-rk-d, rk,
                         CBLAS_SADDR(mzone), A  + rk,              lda,
                                             WT +      (rk+d)*ldw, ldw,
                         CBLAS_SADDR(zone),  A  + rk + (rk+d)*lda, lda );

            cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                         d, n-rk-d, d,
                         CBLAS_SADDR(mzone), Y,                    ldy,
                                             WT + rk + (rk+d)*ldw, ldw,
                         CBLAS_SADDR(zone),  A  + rk + (rk+d)*lda, lda );
        }

        if ( loop && (rk+d < maxrank) ) {
            /*
             * The Q from partial QRCP is stored in the lower part of the matrix,
             * we need to remove it
             */
            ret = LAPACKE_zlaset_work( LAPACK_COL_MAJOR, 'L', d-1, d-1,
                                       0, 0, B + rk*ldb + 1, ldb );
            assert( ret == 0 );

            /* Updating B */
            cblas_ztrsm( CblasColMajor, CblasRight, CblasUpper,
                         CblasNoTrans, CblasNonUnit,
                         d, d,
                         CBLAS_SADDR(zone), A + rk*lda + rk, lda,
                                            B + rk*ldb,      ldb );

            cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                         d, n - (rk+d), d,
                         CBLAS_SADDR(mzone), B +  rk   *ldb,      ldb,
                                             A + (rk+d)*lda + rk, lda,
                         CBLAS_SADDR(zone),  B + (rk+d)*ldb,      ldb );
        }
        rk += d;
    }
    free( jpvt_b );

    if ( d == -1 ) {
        return -1;
    }
    else {
        assert( rk <= maxrank );
        return rk;
    }

    (void)ret;
    (void)refine;
}

/**
 *******************************************************************************
 *
 * @brief Convert a full rank matrix in a low rank matrix, using TQRCP.
 *
 *******************************************************************************
 *
 * @param[in] tol
 *          The tolerance used as a criterion to eliminate information from the
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
core_zge2lr_tqrcp( pastix_fixdbl_t tol, pastix_int_t rklimit,
                   pastix_int_t m, pastix_int_t n,
                   const void *A, pastix_int_t lda,
                   pastix_lrblock_t *Alr )
{
    return core_zge2lr_qr( core_ztqrcp, tol, rklimit,
                           m, n, A, lda, Alr );
}


/**
 *******************************************************************************
 *
 * @brief Add two LR structures A=(-u1) v1^T and B=u2 v2^T into u2 v2^T
 *
 *    u2v2^T - u1v1^T = (u2 u1) (v2 v1)^T
 *    Orthogonalize (u2 u1) = (u2, u1 - u2(u2^T u1)) * (I u2^T u1)
 *                                                     (0    I   )
 *    Compute TQRCP decomposition of (I u2^T u1) * (v2 v1)^T
 *                                   (0    I   )
 *
 *******************************************************************************
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
core_zrradd_tqrcp( const pastix_lr_t *lowrank, pastix_trans_t transA1, const void *alphaptr,
                   pastix_int_t M1, pastix_int_t N1, const pastix_lrblock_t *A,
                   pastix_int_t M2, pastix_int_t N2,       pastix_lrblock_t *B,
                   pastix_int_t offx, pastix_int_t offy)
{
    return core_zrradd_qr( core_ztqrcp, lowrank, transA1, alphaptr,
                           M1, N1, A, M2, N2, B, offx, offy );
}
