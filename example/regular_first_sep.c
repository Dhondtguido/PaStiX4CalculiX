/**
 * @file example/regular_first_sep.c
 *
 * @brief Set by hand the first separator (for regular cube only).
 *
 * @copyright 2015-2017 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @version 6.0.0
 * @author Pierre Ramet
 * @author Mathieu Faverge
 * @author Hastaran Matias
 * @date 2017-05-02
 *
 * @ingroup pastix_examples
 * @code
 *
 */
#include <pastix.h>
#include <spm.h>
#include <lapacke.h>
#include <spm.h>
#include "../common/common.h"

int main (int argc, char **argv)
{
    pastix_data_t  *pastix_data = NULL; /*< Pointer to the storage structure required by pastix */
    pastix_int_t    iparm[IPARM_SIZE];  /*< Integer in/out parameters for pastix                */
    double          dparm[DPARM_SIZE];  /*< Floating in/out parameters for pastix               */
    spm_driver_t    driver;
    char           *filename;
    spmatrix_t     *spm, spm2;
    void           *x, *b, *x0 = NULL;
    size_t          size;
    int             check = 1;
    int             nrhs = 1;
    int             rc    = 0;
    pastix_int_t    nschur;

    /**
     * Initialize parameters to default values
     */
    pastixInitParam( iparm, dparm );

    /**
     * Get options from command line
     */
    pastixGetOptions( argc, argv,
                      iparm, dparm,
                      &check, &driver, &filename );

    if ( driver != SpmDriverLaplacian ){
        fprintf(stderr, "Setting manually the first separator can only be performed with PastixDriverLaplacian driver\n");
        return EXIT_FAILURE;
    }

    if ( (iparm[IPARM_FACTORIZATION] == PastixFactLDLT) ||
         (iparm[IPARM_FACTORIZATION] == PastixFactLDLH) )
    {
        fprintf(stderr, "This types of factorization (LDL^t and LDL^h) are not supported by this example.\n");
        return EXIT_FAILURE;
    }

    /**
     * Read the sparse matrix with the driver
     */
    spm = malloc( sizeof( pastix_spm_t ) );
    spmReadDriver( driver, filename, spm );

    spmPrintInfo( spm, stdout );

    rc = spmCheckAndCorrect( spm, &spm2 );
    if ( rc != 0 ) {
        spmExit( spm );
        *spm = spm2;
    }

    /**
     * Generate a Fake values array if needed for the numerical part
     */
    if ( spm->flttype == PastixPattern ) {
        spmGenFakeValues( spm );
    }

    /**
     * Startup PaStiX
     */
    pastixInit( &pastix_data, MPI_COMM_WORLD, iparm, dparm );

    /**
     * Initialize the schur list with the first third of the unknowns
     */
    {
        pastix_int_t      dim1, dim2, dim3;
        pastix_coeftype_t flttype;
        double            alpha, beta;

        spmParseLaplacianInfo( filename, &flttype, &dim1, &dim2, &dim3, &alpha, &beta );

        (void) flttype;
        (void) alpha;
        (void) beta;
        free(filename);

        if ( (dim1 != dim2) || (dim1 != dim3) ){
            fprintf(stderr, "Grid ordering can be used with a cube only, dimensions are %ld %ld %ld\n", dim1, dim2, dim3);
            return EXIT_FAILURE;
        }

        nschur = dim1 * dim2;

        printf("We put %ld vertices in the schur\n", nschur);

        pastix_int_t i, mp;
        pastix_int_t baseval = spmFindBase(spm);
        pastix_int_t *list = (pastix_int_t*)malloc(nschur * sizeof(pastix_int_t));
        pastix_int_t index;

        /*
         * Laplacian generator is looping over dim3, dim2, and then dim1, thus,
         * the indices of the unkwnowns in the middle plan indexed by:
         * mp = floor(dim3/2)+1 are in the range:
         *  [ dim1 * dim2 * (mp-1), dim1 * dim2 * mp -1 ]
         */
        if ( dim3 > 2 ) {
            mp = dim3 / 2;
            index = dim1 * dim2 * (mp-1);
            for (i=0; i<nschur; i++, index++) {
                list[i] = baseval + index;
            }
            assert( index == dim1 * dim2 * mp );
        }
        else {
            fprintf(stderr, "We do not handle cases with dim3 < 3\n" );
            return EXIT_FAILURE;
        }
        pastixSetSchurUnknownList( pastix_data, nschur, list );
        free(list);
    }

    /**
     * Perform ordering, symbolic factorization, and analyze steps
     */
    pastix_subtask_order( pastix_data, spm, NULL );

    /* Unlock Schur vertices that were used to set manually the first separator */
    {
        pastix_data->schur_n = 0;
        free(pastix_data->schur_list);
        pastix_data->schur_list = NULL;
    }

    pastix_subtask_symbfact( pastix_data );
    pastix_subtask_reordering( pastix_data );
    pastix_subtask_blend( pastix_data );

    /**
     * Normalize A matrix (optional, but recommended for low-rank functionality)
     */
    double normA = spmNorm( PastixFrobeniusNorm, spm );
    spmScalMatrix( 1./normA, spm );

    /**
     * Perform the numerical factorization
     */
    pastix_task_numfact( pastix_data, spm );

    /**
     * Generates the b and x vector such that A * x = b
     * Compute the norms of the initial vectors if checking purpose.
     */
    size = pastix_size_of( spm->flttype ) * spm->n * nrhs;
    x = malloc( size );
    b = malloc( size );

    if ( check )
    {
        if ( check > 1 ) {
            x0 = malloc( size );
        }
        spmGenRHS( SpmRhsRndX, nrhs, spm, x0, spm->n, b, spm->n );
        memcpy( x, b, size );
    }
    else {
        spmGenRHS( SpmRhsRndB, nrhs, spm, NULL, spm->n, x, spm->n );

        /* Apply also normalization to b vectors */
        spmScalVector( spm->flttype, 1./normA, spm->n * nrhs, b, 1 );

        /* Save b for refinement */
        memcpy( b, x, size );
    }

    /**
     * Solve the linear system (and perform the optional refinement)
     */
    pastix_task_solve( pastix_data, nrhs, x, spm->n );
    pastix_task_refine( pastix_data, spm->n, nrhs, b, spm->n, x, spm->n, spm );

    if ( check )
    {
        rc = spmCheckAxb( dparm[DPARM_EPSILON_REFINEMENT], nrhs, spm, x0, spm->n, b, spm->n, x, spm->n );

        if ( x0 ) {
            free( x0 );
        }
    }

    spmExit( spm );
    free( spm );
    free( x );
    free( b );
    pastixFinalize( &pastix_data );

    return rc;
}

/**
 * @endcode
 */
