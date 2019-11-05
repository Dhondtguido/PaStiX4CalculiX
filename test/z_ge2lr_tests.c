/**
 *
 * @file z_ge2lr_tests.c
 *
 * Tests and validate the Xge2lr routine.
 *
 * @copyright 2015-2018 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @version 6.0.1
 * @author Gregoire Pichon
 * @date 2018-07-16
 *
 * @precisions normal z -> z c d s
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
#include "kernels/pastix_lowrank.h"
#include "flops.h"
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

int
z_ge2lr_test( int mode, double tolerance, pastix_int_t rank,
              pastix_int_t m, pastix_int_t n, pastix_int_t lda )
{
    pastix_complex64_t *A = malloc( n * lda * sizeof(pastix_complex64_t) );
    double normA;
    int i, ret, rc = 0;

    /*
     * Generate a matrix of a given rank for the prescribed tolerance
     */
    z_lowrank_genmat( mode, tolerance, rank,
                      m, n, A, lda, &normA );

    fprintf( stdout, "%7s %4s %12s %12s %12s %12s\n",
             "Method", "Rank", "Time", "||A||_f", "||A-UVt||_f",
             "||A-UVt||_f/(||A||_f * eps)" );

    /* Let's test all methods we have */
    for(i=0; i<PastixCompressMethodNbr; i++)
    {
        ret = z_lowrank_check_ge2lr( i, tolerance,
                                     m, n, A, lda, normA,
                                     ge2lrMethods[i][PastixComplex64-2] );
        rc += ret * (1 << i);
    }

    free(A);
    return rc;
}

int main( int argc, char **argv )
{
    (void) argc;
    (void) argv;
    int err = 0;
    int ret;
    pastix_int_t m, r;
    double eps = LAPACKE_dlamch_work('e');
    double tolerance = sqrt(eps);

    for (m=200; m<300; m+=100){
        for (r=0; r <= (m/2); r += ( r + 1 ) ) {
            printf( "   -- Test GE2LR M=N=LDA=%ld R=%ld\n",
                    (long)m, (long)r );

            ret = z_ge2lr_test( 0, tolerance, r, m, m, m );
            PRINT_RES(ret);
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

}
