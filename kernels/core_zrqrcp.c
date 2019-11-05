/**
 *
 * @file core_zrqrcp.c
 *
 * PaStiX Rank-revealing QR kernel beased on randomization technique and partial
 * QR with column pivoting.
 *
 * @copyright 2016-2018 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @version 6.0.0
 * @author Claire Soyez-Martin
 * @date 2018-06-14
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
 * @brief Compute a randomized QR factorization.
 *
 * This kernel implements the algorithm described in:
 * Fast Parallel Randomized QR with Column Pivoting Algorithms for Reliable
 * Low-rank Matrix Approximations. Jianwei Xiao, Ming Gu, and Julien Langou
 *
 * The main difference in this algorithm relies in two points:
 *   1) First, we stop the iterative porcess based on a tolerance criterion
 *   forwarded to the QR with column pivoting kernel, while they have a fixed
 *   number of iterations defined by parameter.
 *   2) Second, we perform an extra PQRCP call on the trailing submatrix to
 *   finalize the computations, while in the paper above they use a spectrum
 *   revealing algorithm to refine the solution.
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
 * @param[in] refine
 *          Enable/disable the extra refinement step that performs an additional
 *          PQRCP on the trailing submatrix.
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
core_zrqrcp( double tol, pastix_int_t maxrank, int refine, pastix_int_t nb,
             pastix_int_t m, pastix_int_t n,
             pastix_complex64_t *A, pastix_int_t lda,
             pastix_int_t *jpvt, pastix_complex64_t *tau,
             pastix_complex64_t *work, pastix_int_t lwork,  double *rwork )
{
    int                 SEED[4] = {26, 67, 52, 197};
    pastix_int_t        j, k, in, itmp, d, ib, loop = 1;
    int                 ret;
    pastix_int_t        b = 24;
    pastix_int_t        p = 8;
    pastix_int_t        bp = b + p;
    pastix_int_t        ldb = bp;
    pastix_int_t        ldo = bp;
    pastix_int_t        size_O = ldo * m;
    pastix_int_t        size_B = ldb * n;
    pastix_int_t       *jpvt_b;
    pastix_int_t        rk, minMN, lwkopt;
    double              tolB = sqrt( (double)(bp) ) * tol;

    pastix_complex64_t *B     = work;
    pastix_complex64_t *tau_b = B + size_B;
    pastix_complex64_t *omega = tau_b + n;
    pastix_complex64_t *subw  = tau_b + n;
    pastix_int_t        sublw = n * nb + pastix_imax( bp, n );
    sublw = pastix_imax( sublw, size_O );

    char trans;
#if defined(PRECISION_c) || defined(PRECISION_z)
    trans = 'C';
#else
    trans = 'T';
#endif

    if ( nb < 0 ) {
        nb = 32;
    }

    lwkopt  = size_B + n + sublw;
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

    minMN = pastix_imin(m, n);
    if ( maxrank < 0 ) {
        maxrank = minMN;
    }
    maxrank = pastix_imin( maxrank, minMN );
    if ( (minMN == 0) || (maxrank == 0) ) {
        return 0;
    }

#if defined(PASTIX_DEBUG_LR)
    B     = malloc( size_B * sizeof(pastix_complex64_t) );
    tau_b = malloc( n      * sizeof(pastix_complex64_t) );
    omega = malloc( size_O * sizeof(pastix_complex64_t) );
    subw  = malloc( sublw  * sizeof(pastix_complex64_t) );
#endif

    jpvt_b = malloc( n * sizeof(pastix_int_t) );
    for (j=0; j<n; j++) jpvt[j] = j;

    /* Computation of the Gaussian matrix */
    LAPACKE_zlarnv_work(3, SEED, size_O, omega);
    cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                 bp, n, m,
                 CBLAS_SADDR(zone),  omega, ldo,
                                     A, lda,
                 CBLAS_SADDR(zzero), B, ldb );

    rk = 0;
    while ( (rk < maxrank) && loop )
    {
        ib = pastix_imin( b, maxrank-rk );
        d = core_zpqrcp( tolB, ib, 1, nb,
                         bp, n-rk,
                         B + rk*ldb, ldb,
                         jpvt_b + rk, tau_b,
                         subw, sublw, /* >= (n*nb)+max(bp, n) */
                         rwork );     /* >=  2*n */

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

        /* Updating jpvt and A */
        for (j = rk; j < rk + d; j++) {
            if (jpvt_b[j] >= 0) {
                k = j;
                in = jpvt_b[k] + rk;

                /* Mark as done */
                jpvt_b[k] = - jpvt_b[k] - 1;

                while( jpvt_b[in] >= 0 ) {

                    if (k != in) {
                        cblas_zswap( m, A + k  * lda, 1,
                                        A + in * lda, 1 );

                        itmp     = jpvt[k];
                        jpvt[k]  = jpvt[in];
                        jpvt[in] = itmp;
                    }
                    itmp = jpvt_b[in];
                    jpvt_b[in] = - jpvt_b[in] - 1;
                    k = in;
                    in = itmp + rk;
                }
            }
        }

        /*
         * Factorize d columns of A without pivoting
         */
        ret = LAPACKE_zgeqrf_work( LAPACK_COL_MAJOR, m-rk, d,
                                   A + rk*lda + rk, lda, tau + rk,
                                   subw, sublw );
        assert(ret == 0);

        if ( rk+d < n ) {

            /*
             * Update trailing submatrix: A <- Q^h A
             */
            ret = LAPACKE_zunmqr_work( LAPACK_COL_MAJOR, 'L', trans,
                                       m-rk, n-rk-d, d,
                                       A +  rk   *lda + rk, lda, tau + rk,
                                       A + (rk+d)*lda + rk, lda,
                                       subw, sublw );
            assert(ret == 0);

            /*
             * The Q from partial QRCP is stored in the lower part of the matrix,
             * we need to remove it
             */
            ret = LAPACKE_zlaset_work( LAPACK_COL_MAJOR, 'L', d-1, d-1,
                                       0, 0, B + rk*ldb + 1, ldb );
            assert( ret == 0 );

            /*
             * Updating B
             */
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

    d = 0;
    if ( refine && !loop && (rk < maxrank) ) {
        /*
         * Apply a Rank-revealing QR on the trailing submatrix to get the last
         * columns
         */
        d = core_zpqrcp( tol, maxrank-rk, 0, nb,
                         m-rk, n-rk,
                         A + rk * lda + rk, lda,
                         jpvt_b, tau + rk,
                         work, lwork, rwork );

        /* Updating jpvt and A */
        for (j=0; j<d; j++) {
            if (jpvt_b[j] >= 0) {
                k = j;
                in = jpvt_b[k];

                /* Mark as done */
                jpvt_b[k] = - jpvt_b[k] - 1;

                while( jpvt_b[in] >= 0 ) {
                    if (k != in) {
                        /* Swap columns in first rows */
                        cblas_zswap( rk, A + (rk + k ) * lda, 1,
                                         A + (rk + in) * lda, 1 );

                        itmp          = jpvt[rk + k];
                        jpvt[rk + k]  = jpvt[rk + in];
                        jpvt[rk + in] = itmp;
                    }
                    itmp = jpvt_b[in];
                    jpvt_b[in] = - jpvt_b[in] - 1;
                    k = in;
                    in = itmp;
                }
            }
        }
    }
    free( jpvt_b );

#if defined(PASTIX_DEBUG_LR)
    free( B     );
    free( tau_b );
    free( omega );
    free( subw  );
#endif

    if ( d == -1 ) {
        return -1;
    }
    else if ( rk < maxrank ) {
        return pastix_imin( maxrank, rk+d );
    }
    else {
        return rk;
    }
    (void)ret;
}

/**
 *******************************************************************************
 *
 * @brief Convert a full rank matrix in a low rank matrix, using RQRCP.
 *
 *******************************************************************************
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
core_zge2lr_rqrcp( pastix_fixdbl_t tol, pastix_int_t rklimit,
                   pastix_int_t m, pastix_int_t n,
                   const void *A, pastix_int_t lda,
                   pastix_lrblock_t *Alr )
{
    return core_zge2lr_qr( core_zrqrcp, tol, rklimit,
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
 *    Compute RQRCP decomposition of (I u2^T u1) * (v2 v1)^T
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
core_zrradd_rqrcp( const pastix_lr_t *lowrank, pastix_trans_t transA1, const void *alphaptr,
                   pastix_int_t M1, pastix_int_t N1, const pastix_lrblock_t *A,
                   pastix_int_t M2, pastix_int_t N2,       pastix_lrblock_t *B,
                   pastix_int_t offx, pastix_int_t offy)
{
    return core_zrradd_qr( core_zrqrcp, lowrank, transA1, alphaptr,
                           M1, N1, A, M2, N2, B, offx, offy );
}
