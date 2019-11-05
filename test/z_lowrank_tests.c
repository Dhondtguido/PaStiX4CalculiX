/**
 *
 * @file z_lowrank_tests.c
 *
 * Test functions for the low-rank kernels.
 *
 * @copyright 2015-2018 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @version 6.0.1
 * @author Gregoire Pichon
 * @author Mathieu Faverge
 * @date 2018-07-16
 *
 * @precisions normal z -> z c d s
 *
 **/
#include <stdio.h>
#include <assert.h>
#include <pastix.h>
#include "common/common.h"
#include <lapacke.h>
#include "blend/solver.h"
#include "kernels/pastix_zcores.h"
#include "kernels/pastix_zlrcores.h"
#include "kernels/pastix_lowrank.h"
#include "z_tests.h"

static pastix_complex64_t mzone = -1.0;
static pastix_complex64_t zone  =  1.0;

/* Initialize a basic stucture for kernels needing it */
pastix_lr_t z_lowrank = {
    .compress_when       = PastixCompressWhenEnd,
    .compress_method     = PastixCompressMethodPQRCP,
    .compress_min_width  = 0,
    .compress_min_height = 0,
    .tolerance  = 0.,
    .core_ge2lr = core_zge2lr_pqrcp,
    .core_rradd = core_zrradd_pqrcp,
};

static int ISEED[4] = { 42, 15, 314, 666 };   /* initial seed for zlarnv() */

int
z_lowrank_genmat( int mode, double tolerance, pastix_int_t rank,
                  pastix_int_t m, pastix_int_t n,
                  pastix_complex64_t *A, pastix_int_t lda,
                  double             *normA )
{
    pastix_int_t minMN    = pastix_imin(m, n);
    double       rcond    = (double) minMN;
    double       dmax     = 1.0;
    pastix_complex64_t *work;
    double *S;

    if (m < 0) {
        fprintf(stderr, "Invalid m parameter\n");
        return -4;
    }
    if (n < 0) {
        fprintf(stderr, "Invalid n parameter\n");
        return -5;
    }
    if (lda < m) {
        fprintf(stderr, "Invalid lda parameter\n");
        return -6;
    }
    if (rank > pastix_imin( m, n )) {
        fprintf(stderr, "Invalid rank parameter\n");
        return -3;
    }

    S    = malloc(minMN * sizeof(double));
    work = malloc(3 * pastix_imax(m, n)* sizeof(pastix_complex64_t));

    if ( (!S) || (!work) ) {
        fprintf(stderr, "Out of Memory\n");
        free(S); free(work);
        return -2;
    }

    /*
     * Choose alpha such that alpha^rank = tolerance
     */
    if (mode == 0) {
        pastix_int_t i;
        double alpha = exp(log(tolerance) / rank);

        S[0] = 1;

        if (rank == 0) {
            S[0] = 0.;
        }
        for (i=1; i<minMN; i++){
            S[i] = S[i-1] * alpha;
        }
    }

    /* Initialize A */
    LAPACKE_zlatms_work( LAPACK_COL_MAJOR, m, n,
                         'U', ISEED,
                         'N', S, mode, rcond,
                         dmax, m, n,
                         'N', A, lda, work );

    *normA = LAPACKE_zlange_work( LAPACK_COL_MAJOR, 'f', m, n,
                                  A, lda, NULL );

    free(S); free(work);
    return 0;
}

int
z_lowrank_check_ge2lr( pastix_compress_method_t method, double tolerance,
                       pastix_int_t m, pastix_int_t n,
                       pastix_complex64_t *A, pastix_int_t lda,
                       double normA,
                       fct_ge2lr_t core_ge2lr )
{
    pastix_lrblock_t    lrA;
    pastix_complex64_t *A2;
    pastix_int_t minMN = pastix_imin(m, n);
    Clock timer;
    double resid, norm_residual;
    double total_timer = 0.;
    pastix_int_t i, rankA, nb_tests = 1;

    if (m < 0) {
        fprintf(stderr, "Invalid m parameter\n");
        return -4;
    }
    if (n < 0) {
        fprintf(stderr, "Invalid n parameter\n");
        return -5;
    }
    if (lda < m) {
        fprintf(stderr, "Invalid lda parameter\n");
        return -6;
    }
    A2 = malloc(n * lda * sizeof(pastix_complex64_t));

    for (i = 0 ; i < nb_tests ; i++) {
        /*
         * Compress and then uncompress
         */
        clockStart(timer);
        core_ge2lr( tolerance, minMN,
                    m, n, A, lda,
                    &lrA );
        clockStop(timer);
        total_timer += clockVal(timer);

        /*
         * Let's check the result
         */
        core_zlr2ge( PastixNoTrans, m, n,
                     &lrA, A2, lda );

        core_zgeadd( PastixNoTrans, m, n,
                     -1., A,  lda,
                      1., A2, lda );

        norm_residual = LAPACKE_zlange_work( LAPACK_COL_MAJOR, 'f', m, n,
                                             A2, lda, NULL );

        if ( normA > 0. ) {
            resid = norm_residual / ( tolerance * normA );
        }
        else {
            resid = norm_residual;
        }

        rankA = lrA.rk;
        core_zlrfree( &lrA );
    }

    total_timer /= nb_tests;

    fprintf( stdout, "%7s %4d %e %e %e %e %7s\n",
             compmeth_shnames[method], (int)rankA, total_timer, normA, norm_residual, resid,
             (resid > 10.) ? "FAILED" : "SUCCESS" );

    free(A2);

    if ( resid > 10. ) {
        return 1;
    }
    else {
        return 0;
    }
}

int
z_lowrank_check_rradd( pastix_compress_method_t method, double tolerance,
                       pastix_int_t offx, pastix_int_t offy,
                       pastix_int_t mA, pastix_int_t nA,
                       const pastix_lrblock_t *lrA,
                       pastix_int_t mB, pastix_int_t nB,
                       const pastix_lrblock_t *lrB,
                       const pastix_complex64_t *Cfr, pastix_int_t ldc, double normCfr,
                       fct_rradd_t core_rradd )
{
    pastix_complex64_t *Clr;
    double norm_diff, res;
    Clock timer;
    pastix_int_t rkCmax;
    pastix_int_t rankmax  = core_get_rklimit(mB, nB);
    int          rc = 0;
    pastix_lrblock_t lrC;

    rkCmax = lrA->rk + lrB->rk;

    if ( (lrA->rk == -1) ||
         (lrB->rk == -1) )
    {
        printf("Operation not supported\n");
        return 0;
    }

    /* Init lrC */
    memset( &lrC, 0, sizeof(pastix_lrblock_t) );

    /* Backup B into C */
    core_zlrcpy( NULL, PastixNoTrans, 1.,
                 mB, nB, lrB, mB, mB, &lrC, 0, 0 );

    /* Perform C = B - A in LR format */
    z_lowrank.tolerance = tolerance;
    clockStart(timer);
    core_rradd( &z_lowrank, PastixNoTrans, &mzone,
                mA, nA, lrA,
                mB, nB, &lrC,
                offx, offy );
    clockStop(timer);

    /*
     * Check || (A+B) - c( c(A)+c(B) ) || < tol * || A+B ||
     */
     Clr = malloc( mB * nB * sizeof(pastix_complex64_t) );

    /* Uncompress the low-rank sum */
    core_zlr2ge( PastixNoTrans, mB, nB,
                 &lrC, Clr, mB );

    /* Compute the diff */
    core_zgeadd( PastixNoTrans, mB, nB,
                 -1., Cfr, ldc,
                  1., Clr, mB );

    norm_diff = LAPACKE_zlange_work( LAPACK_COL_MAJOR, 'f', mB, nB,
                                     Clr, mB, NULL );

    if ( (lrA->rk != 0) || (lrB->rk != 0) ) {
        res = norm_diff / ( tolerance * (normCfr) );
    }
    else {
        res = norm_diff;
    }

    free(Clr);

    /* Check the correctness of the result */
    if ( res > 10.0 ) {
        rc += 1;
    }

    /* Check that final matrix is not full rank, we should have exited before */
    if ( (lrC.rk == -1) && (rkCmax <= rankmax) ) {
        rc += 2;
    }

    /* Check that final rank does not exceed the sum of the ranks */
    if ( lrC.rk > rkCmax ) {
        rc += 4;
    }

    /* Check that final rank is larger than the minimal rank (rA, rB, abs(rA-rb)) */
    if ( (lrC.rk != -1) && (lrC.rk < pastix_imin( pastix_imin( lrA->rk, lrB->rk ), abs(lrA->rk - lrB->rk) )) )
    {
        rc += 8;
    }

    fprintf( stdout, "%7s %4d %e %e %e %e ",
             compmeth_shnames[method], (int)lrC.rk, clockVal(timer), normCfr, norm_diff, res );

    if ( rc == 0 ) {
        fprintf( stdout, "SUCCESS\n" );
    }
    else {
        fprintf( stdout, "FAILED(%d)\n", rc );
    }

    core_zlrfree( &lrC );
    return (rc > 0) ? 1 : 0;
}

int
z_lowrank_check_lrmm( pastix_compress_method_t method, double tolerance,
                      pastix_int_t offx, pastix_int_t offy,
                      pastix_int_t m, pastix_int_t n, pastix_int_t k,
                      const pastix_lrblock_t *lrA,
                      const pastix_lrblock_t *lrB,
                      pastix_int_t Cm, pastix_int_t Cn,
                      const pastix_lrblock_t *lrC,
                      const pastix_complex64_t *Cfr, pastix_int_t ldc,
                      double normCfr,
                      pastix_lr_t *lowrank )
{
    pastix_lrblock_t lrC2;
    pastix_complex64_t *Clr;
    double norm_diff, res;
    Clock timer;
    int rc = 0;

    /* Init lrC */
    memset( &lrC2, 0, sizeof(pastix_lrblock_t) );

    /* Backup C into C2 */
    core_zlrcpy( NULL, PastixNoTrans, 1.,
                 Cm, Cn, lrC, Cm, Cn, &lrC2, 0, 0 );

    /* Compute the low-rank matrix-matrix */
    {
        pastix_atomic_lock_t lock = PASTIX_ATOMIC_UNLOCKED;
        core_zlrmm_t zlrmm_params;
        zlrmm_params.lowrank = lowrank;
        zlrmm_params.transA  = PastixNoTrans;
        zlrmm_params.transB  = PastixConjTrans;
        zlrmm_params.M       = m;
        zlrmm_params.N       = n;
        zlrmm_params.K       = k;
        zlrmm_params.Cm      = Cm;
        zlrmm_params.Cn      = Cn;
        zlrmm_params.offx    = offx;
        zlrmm_params.offy    = offy;
        zlrmm_params.alpha   = mzone;
        zlrmm_params.A       = lrA;
        zlrmm_params.B       = lrB;
        zlrmm_params.beta    = zone;
        zlrmm_params.C       = &lrC2;
        zlrmm_params.work    = NULL;
        zlrmm_params.lwork   = -1;
        zlrmm_params.lwused  = -1;
        zlrmm_params.lock    = &lock;

        clockStart(timer);
        core_zlrmm( &zlrmm_params );
        clockStop(timer);
    }

    /*
     * Check || (A+B) - (c(A)+c(B)) || < tol * || A+B ||
     * What we really want is || (A+B) - c(A+B) || < tol * || A+B ||
     */
    Clr = malloc( Cm * Cn * sizeof(pastix_complex64_t) );

    /* Uncompress the low-rank sum */
    core_zlr2ge( PastixNoTrans, Cm, Cn,
                 &lrC2, Clr, Cm );

    /* Compute the diff */
    core_zgeadd( PastixNoTrans, Cm, Cn,
                 -1., Cfr, ldc,
                  1., Clr, Cm );

    norm_diff = LAPACKE_zlange_work( LAPACK_COL_MAJOR, 'f', Cm, Cn,
                                     Clr, Cm, NULL );

    if ( (lrC->rk != 0.0) && ((lrA->rk + lrB->rk) != 0) ) {
        res = norm_diff / ( tolerance * normCfr );
    }
    else {
        res = norm_diff;
    }

    fprintf( stdout, "%7s %4d %e %e %e %e ",
             compmeth_shnames[method], (int)lrC2.rk, clockVal(timer), normCfr, norm_diff, res );

    free(Clr);
    core_zlrfree( &lrC2 );

    /* Check the correctness of the result */
    if ( res > 10.0 ) {
        rc += 1;
    }

    if ( rc == 0 ) {
        fprintf( stdout, "SUCCESS\n" );
    }
    else {
        fprintf( stdout, "FAILED(%d)\n", rc );
    }

    return rc;
}
