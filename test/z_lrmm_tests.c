/**
 *
 * @file z_lrmm_tests.c
 *
 * Tests and validate the core_zlrmm() routine.
 *
 * @copyright 2015-2018 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @version 6.0.1
 * @author Gregoire Pichon
 * @date 2018-07-16
 *
 * @precisions normal z -> c d s
 *
 **/
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <pastix.h>
#include "common/common.h"
#include <lapacke.h>
#include <cblas.h>
#include "blend/solver.h"
#include "kernels/pastix_zcores.h"
#include "kernels/pastix_zlrcores.h"
#include "z_tests.h"

#define PRINT_RES(_ret_)                        \
    if(_ret_ == -1) {                           \
        printf("UNDEFINED\n");                  \
    }                                           \
    else if(_ret_ > 0) {                        \
        printf("FAILED(%d)\n", _ret_);          \
        err++;                                  \
    }                                           \
    else {                                      \
        printf("SUCCESS\n");                    \
    }

int main( int argc, char **argv )
{
    fct_ge2lr_t core_ge2lr = ge2lrMethods[PastixCompressMethodPQRCP][PastixComplex64-2];
    int err = 0;
    int ret = 0;
    int mode = 0;
    pastix_int_t m, n, k, Cm, Cn, offx, offy;
    pastix_int_t lda, ldb, ldc;
    double       eps = LAPACKE_dlamch_work('e');
    double       tolerance = sqrt(eps);
    pastix_complex64_t *A, *B, *C;
    double              normA, normB, normC;
    int ranks[3], rA, rB, rC, r, s, i, j, l, meth;
    pastix_lrblock_t lrA, lrB, lrC;

    pastix_complex64_t mzone = -1.0;
    pastix_complex64_t zone  = 1.0;

    core_get_rklimit = core_get_rklimit_max;

    for (s=100; s<=200; s = 2*s) {
        ranks[0] = s + 1;
        ranks[1] = 16;
        ranks[2] = 2;

        m = s / 2;
        n = s / 4;
        k = s / 6;

        offx = 1;
        offy = 2;

        Cm = s;
        Cn = s;

        /* Matrix A */
        for (i=0; i<3; i++) {
            rA = pastix_imin( m, k ) / ranks[i];
            lda = m;

            A = malloc( lda * k * sizeof( pastix_complex64_t ) );
            z_lowrank_genmat( mode, tolerance, rA,
                              m, k, A, lda, &normA );
            core_ge2lr( tolerance, pastix_imin( m, k ),
                        m, k, A, lda, &lrA );
            core_zlr2ge( PastixNoTrans, m, k,
                         &lrA, A, lda );

            /* Matrix B */
            for (j=0; j<3; j++) {
                rB = pastix_imin( n, k ) / ranks[j];
                ldb = n;

                B = malloc( ldb * k * sizeof( pastix_complex64_t ) );
                z_lowrank_genmat( mode, tolerance, rB,
                                  n, k, B, ldb, &normB );
                core_ge2lr( tolerance, pastix_imin( n, k ),
                            n, k, B, ldb, &lrB );
                core_zlr2ge( PastixNoTrans, n, k,
                             &lrB, B, ldb );

                /* Matrix C */
                for (l=0; l<3; l++) {
                    rC = pastix_imin( Cm, Cn ) / ranks[l];
                    ldc = Cm;

                    C = malloc( ldc * Cn * sizeof( pastix_complex64_t ) );
                    z_lowrank_genmat( mode, tolerance, rC,
                                      Cm, Cn, C, ldc, &normC );
                    core_ge2lr( tolerance, pastix_imin( Cm, Cn ),
                                Cm, Cn, C, ldc, &lrC );
                    core_zlr2ge( PastixNoTrans, Cm, Cn,
                                 &lrC, C, ldc );

                    printf( "  -- Test LRMM Cm=%ld, Cn=%ld, m=%ld, n=%ld, k=%ld, rA=%ld, rB=%ld, rC=%ld\n",
                            (long)Cm, (long)Cn, (long)m, (long)n, (long)k, (long)lrA.rk, (long)lrB.rk, (long)lrC.rk );

                    /* Compute the full rank GEMM */
                    cblas_zgemm( CblasColMajor, CblasNoTrans, CblasConjTrans,
                                 m, n, k,
                                 CBLAS_SADDR( mzone ), A, lda,
                                                       B, ldb,
                                 CBLAS_SADDR( zone ),  C + ldc * offy + offx, ldc );

                    normC = LAPACKE_zlange_work( LAPACK_COL_MAJOR, 'f', Cm, Cn,
                                                 C, ldc, NULL );

                    fprintf( stdout, "%7s %4s %12s %12s %12s %12s\n",
                             "Method", "Rank", "Time", "||C||_f", "||c(C)-C||_f",
                             "||c(C)-C||_f/(||C||_f * eps)" );

                    ret = 0;

                    /* Let's test all methods we have */
                    for(meth=0; meth<PastixCompressMethodNbr; meth++)
                    {
                        z_lowrank.compress_method = meth;
                        z_lowrank.core_ge2lr = ge2lrMethods[meth][PastixComplex64-2];
                        z_lowrank.core_rradd = rraddMethods[meth][PastixComplex64-2];

                        r = z_lowrank_check_lrmm( meth, tolerance,
                                                  offx, offy,
                                                  m, n, k, &lrA, &lrB,
                                                  Cm, Cn, &lrC,
                                                  C, ldc, normC,
                                                  &z_lowrank );

                        ret += r * (1 << meth);
                    }

                    core_zlrfree( &lrC );
                    free( C );

                    PRINT_RES( ret );
                }

                core_zlrfree( &lrB );
                free( B );
            }
            core_zlrfree( &lrA );
            free( A );
        }
    }

    if( err == 0 ) {
        printf(" -- All tests PASSED --\n");
        return EXIT_SUCCESS;
    }
    else
    {
        printf(" -- %d tests FAILED --\n", err);
        return EXIT_FAILURE;
    }

    (void) argc;
    (void) argv;
}
