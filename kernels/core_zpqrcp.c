/**
 *
 * @file core_zpqrcp.c
 *
 * PaStiX implementation of the partial rank-revealing QR with column pivoting
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
 * @brief Compute a rank-reavealing QR factorization.
 *
 * This routine is originated from the LAPACK kernels zgeqp3/zlaqps and was
 * modified by A. Buttari for MUMPS-BLR.
 * In this version the stopping criterion is based on the frobeniux norm of the
 * residual, and not on the estimate of the two-norm making it more
 * restrictive. Thus, the returned ranks are larger, but this gives a better
 * accuracy.
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
 * @param[in] full_update
 *          If true, all the trailing submatrix is updated, even if maxrank is
 *          reached.
 *          If false, the trailing submatrix is not updated as soon as it is not
 *          worth it. (Unused for now but kept to match API of RQRCP and TQRCP)
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
core_zpqrcp( double tol, pastix_int_t maxrank, int full_update, pastix_int_t nb,
             pastix_int_t m, pastix_int_t n,
             pastix_complex64_t *A, pastix_int_t lda,
             pastix_int_t *jpvt, pastix_complex64_t *tau,
             pastix_complex64_t *work, pastix_int_t lwork,  double *rwork )
{
    pastix_int_t minMN, ldf, lwkopt;
    pastix_int_t j, k, jb, itemp, lsticc, pvt;
    double temp, temp2, machine_prec, residual;
    pastix_complex64_t akk, *auxv, *f;

    /* Partial (VN1) and exact (VN2) column norms */
    double *VN1, *VN2;

    /* Number or rows of A that have been factorized */
    pastix_int_t offset = 0;

    /* Rank */
    pastix_int_t rk = 0;

    if (nb < 0) {
        nb = 32;
    }

    lwkopt = n * nb + pastix_imax(m, n);
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
    maxrank = pastix_imin( minMN, maxrank );
    if ( (minMN == 0) || (maxrank == 0) ) {
        return 0;
    }

    VN1 = rwork;
    VN2 = rwork + n;

    auxv = work;
    f    = work + pastix_imax(m, n);
    ldf  = n;

    /*
     * Initialize partial column norms. The first N elements of work store the
     * exact column norms.
     */
    for (j=0; j<n; j++){
        VN1[j]  = cblas_dznrm2(m, A + j * lda, 1);
        VN2[j]  = VN1[j];
        jpvt[j] = j;
    }

    offset = 0;
    machine_prec = sqrt(LAPACKE_dlamch_work('e'));
    rk = 0;

    while ( rk < maxrank ) {
        /* jb equivalent to kb in LAPACK xLAQPS: maximum number of columns to factorize */
        jb     = pastix_imin(nb, maxrank-offset);
        lsticc = 0;

        /* Factorize as many columns as possible */
        for ( k=0; k<jb; k++ ) {

            rk = offset + k;

            assert( rk < maxrank );

            pvt = rk + cblas_idamax( n-rk, VN1 + rk, 1 );

            /*
             * The selected pivot is below the threshold, we check if we exit
             * now or we still need to compute it to refine the precision.
             */
            if ( (VN1[pvt] == 0.) || (VN1[pvt] < tol) ) {
                residual = cblas_dnrm2( n-rk, VN1 + rk, 1 );
                if ( (residual == 0.) || (residual < tol) ) {
                    return rk;
                }
            }

            /*
             * Pivot is not within the current column: we swap
             */
            if ( pvt != rk ) {
                assert( pvt < n );
                cblas_zswap( m, A + pvt * lda, 1,
                                A + rk  * lda, 1 );
                cblas_zswap( k, f + (pvt-offset), ldf,
                                f + k,            ldf );

                itemp     = jpvt[pvt];
                jpvt[pvt] = jpvt[rk];
                jpvt[rk]  = itemp;
                VN1[pvt]  = VN1[rk];
                VN2[pvt]  = VN2[rk];
            }

            /*
             * Apply previous Householder reflectors to the column K
             * A(RK:M,RK) := A(RK:M,RK) - A(RK:M,OFFSET+1:RK-1)*F(K,1:K-1)**H
             */
            if ( k > 0 ) {
                assert( (rk < n) && (rk < m) );

#if defined(PRECISION_c) || defined(PRECISION_z)
                cblas_zgemm( CblasColMajor, CblasNoTrans, CblasConjTrans, m-rk, 1, k,
                             CBLAS_SADDR(mzone), A + offset * lda + rk, lda,
                                                 f +                k,  ldf,
                             CBLAS_SADDR(zone),  A + rk     * lda + rk, lda );
#else
                cblas_zgemv( CblasColMajor, CblasNoTrans, m-rk, k,
                             CBLAS_SADDR(mzone), A + offset * lda + rk, lda,
                                                 f +                k,  ldf,
                             CBLAS_SADDR(zone),  A + rk     * lda + rk, 1 );
#endif
            }

            /*
             * Generate elementary reflector H(k).
             */
            if ((rk+1) < m) {
                LAPACKE_zlarfg(m-rk, A + rk * lda + rk, A + rk * lda + (rk+1), 1, tau + rk);
            }
            else{
                LAPACKE_zlarfg(1,    A + rk * lda + rk, A + rk * lda + rk,     1, tau + rk);
            }

            akk = A[rk * lda + rk];
            A[rk * lda + rk] = zone;

            /*
             * Compute Kth column of F:
             * F(K+1:N,K) := tau(K)*A(RK:M,K+1:N)**H*A(RK:M,K).
             */
            if ((rk+1) < n) {
                pastix_complex64_t alpha = tau[rk];
                cblas_zgemv( CblasColMajor, CblasConjTrans, m-rk, n-rk-1,
                             CBLAS_SADDR(alpha), A + (rk+1) * lda + rk,    lda,
                                                 A +  rk    * lda + rk,    1,
                             CBLAS_SADDR(zzero), f +  k     * ldf + k + 1, 1 );
            }

            /*
             * Padding F(1:K,K) with zeros.
             */
            memset( f + k * ldf, 0, k * sizeof( pastix_complex64_t ) );

            /*
             * Incremental updating of F:
             * F(1:N,K) := F(1:N-OFFSET,K) - tau(RK)*F(1:N,1:K-1)*A(RK:M,OFFSET+1:RK-1)**H*A(RK:M,RK).
             */
            if (k > 0) {
                pastix_complex64_t alpha = -tau[rk];
                cblas_zgemv( CblasColMajor, CblasConjTrans, m-rk, k,
                             CBLAS_SADDR(alpha), A + offset * lda + rk, lda,
                                                 A + rk     * lda + rk, 1,
                             CBLAS_SADDR(zzero), auxv,                  1 );

                cblas_zgemv( CblasColMajor, CblasNoTrans, n-offset, k,
                             CBLAS_SADDR(zone), f,           ldf,
                                                auxv,        1,
                             CBLAS_SADDR(zone), f + k * ldf, 1);
            }

            /*
             * Update the current row of A:
             * A(RK,RK+1:N) := A(RK,RK+1:N) - A(RK,OFFSET+1:RK)*F(K+1:N,1:K)**H.
             */
            if ((rk+1) < n) {
#if defined(PRECISION_c) || defined(PRECISION_z)
                cblas_zgemm( CblasColMajor, CblasNoTrans, CblasConjTrans,
                             1, n-rk-1, k+1,
                             CBLAS_SADDR(mzone), A + (offset) * lda + rk,    lda,
                                                 f +                  (k+1), ldf,
                             CBLAS_SADDR(zone),  A + (rk + 1) * lda + rk,    lda );
#else
                cblas_zgemv( CblasColMajor, CblasNoTrans, n-rk-1, k+1,
                             CBLAS_SADDR(mzone), f +                  (k+1), ldf,
                                                 A + (offset) * lda + rk,    lda,
                             CBLAS_SADDR(zone),  A + (rk + 1) * lda + rk,    lda );
#endif
            }

            /*
             * Update partial column norms.
             */
            for (j=rk+1; j<n; j++) {
                if (VN1[j] != 0.0) {
                    /*
                     * NOTE: The following 4 lines follow from the analysis in
                     * Lapack Working Note 176.
                     */
                    temp  = cabs( A[j * lda + rk] ) / VN1[j];
                    temp2 = (1.0 + temp) * (1.0 - temp);
                    temp  = (temp2 > 0.0) ? temp2 : 0.0;

                    temp2 = temp * ((VN1[j] / VN2[j]) * ( VN1[j] / VN2[j]));
                    if (temp2 < machine_prec){
                        VN2[j] = (double)lsticc;
                        lsticc = j;
                    }
                    else{
                        VN1[j] = VN1[j] * sqrt(temp);
                    }
                }
            }

            A[rk * lda + rk] = akk;

            if (lsticc != 0) {
                k++;
                break;
            }
        }

        /* One additional reflector has been computed */
        rk++;

        /*
         * Apply the block reflector to the rest of the matrix:
         * A(RK+1:M,RK+1:N) := A(RK+1:M,RK+1:N) -
         * A(RK+1:M,OFFSET+1:RK)*F(K+1:N-OFFSET,1:K)**H.
         */
        if ( rk < n )
        {
            cblas_zgemm( CblasColMajor, CblasNoTrans, CblasConjTrans,
                         m-rk, n-rk, k,
                         CBLAS_SADDR(mzone), A + offset * lda + rk, lda,
                                             f +                k,  ldf,
                         CBLAS_SADDR(zone),  A + rk     * lda + rk, lda );
        }

        /* Recomputation of difficult columns. */
        while (lsticc > 0) {
            assert(lsticc < n);
            itemp = (pastix_int_t) (VN2[lsticc]);

            VN1[lsticc] = cblas_dznrm2(m-rk, A + lsticc * lda + rk, 1 );

            /*
             * NOTE: The computation of VN1( LSTICC ) relies on the fact that
             * SNRM2 does not fail on vectors with norm below the value of
             * SQRT(DLAMCH('S'))
             */
            VN2[lsticc] = VN1[lsticc];
            lsticc = itemp;
        }

        offset = rk;
    }

    /* We reached maxrank, so we check if the threshold is met or not */
    residual = cblas_dnrm2( n-rk, VN1 + rk, 1 );
    if ( (tol < 0) || ( (residual == 0.) || (residual < tol) ) ) {
        assert( rk == maxrank );
        return rk;
    }
    else {
        return -1;
    }

    (void)full_update;
}

/**
 *******************************************************************************
 *
 * @brief Convert a full rank matrix in a low rank matrix, using PQRCP.
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
core_zge2lr_pqrcp( pastix_fixdbl_t tol, pastix_int_t rklimit,
                   pastix_int_t m, pastix_int_t n,
                   const void *A, pastix_int_t lda,
                   pastix_lrblock_t *Alr )
{
    return core_zge2lr_qr( core_zpqrcp, tol, rklimit,
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
 *    Compute PQRCP decomposition of (I u2^T u1) * (v2 v1)^T
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
core_zrradd_pqrcp( const pastix_lr_t *lowrank, pastix_trans_t transA1, const void *alphaptr,
                   pastix_int_t M1, pastix_int_t N1, const pastix_lrblock_t *A,
                   pastix_int_t M2, pastix_int_t N2,       pastix_lrblock_t *B,
                   pastix_int_t offx, pastix_int_t offy)
{
    return core_zrradd_qr( core_zpqrcp, lowrank, transA1, alphaptr,
                           M1, N1, A, M2, N2, B, offx, offy );
}
