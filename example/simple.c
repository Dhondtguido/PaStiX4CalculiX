/**
 * @file simple.c
 *
 * @brief A simple example that reads the matrix and then runs pastix in one call.
 *
 * @copyright 2015-2018 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @version 6.0.1
 * @author Hastaran Matias
 * @date 2018-07-16
 *
 * @ingroup pastix_examples
 * @code
 *
 */
#include <pastix.h>
#include <spm.h>

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
    int             nrhs  = 1;
    int             rc    = 0;

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

    /**
     * Read the sparse matrix with the driver
     */
    spm = malloc( sizeof( spmatrix_t ) );
    spmReadDriver( driver, filename, spm );
    free( filename );

    spmPrintInfo( spm, stdout );

    rc = spmCheckAndCorrect( spm, &spm2 );
    if ( rc != 0 ) {
        spmExit( spm );
        *spm = spm2;
    }

    /**
     * Generate a Fake values array if needed for the numerical part
     */
    if ( spm->flttype == SpmPattern ) {
        spmGenFakeValues( spm );
    }

    /**
     * Startup PaStiX
     */
    pastixInit( &pastix_data, MPI_COMM_WORLD, iparm, dparm );

    /**
     * Perform ordering, symbolic factorization, and analyze steps
     */
    pastix_task_analyze( pastix_data, spm );

    /**
     * Normalize A matrix (optional, but recommended for low-rank functionality)
     */
//    double normA = spmNorm( SpmFrobeniusNorm, spm );
//    spmScalMatrix( 1./normA, spm );
    size = pastix_size_of( spm->flttype ) * spm->n * nrhs;
    x = malloc( size );
    b = malloc( size );
printf("size: %i\n", size);
  FILE* file = fopen ("/ya/ya127/ya12797/w/Solver_dev/Solve_dev/testdaten/job6Data/b.mtx", "r");
  int i = 0;
  while (!feof (file))
    {  
      fscanf (file, "%lf", &(((double*)x)[i]));      
 //     printf ("%lf\n", ((double*)x)[i]);      
      i++;
if(i == spm->n)
    break;
    }
  fclose (file);    

    memcpy( b, x, size );



    /**
     * Perform the numerical factorization
     */
    pastix_task_numfact( pastix_data, spm );

    /**
     * Generates the b and x vector such that A * x = b
     * Compute the norms of the initial vectors if checking purpose.
     */

/*
    if ( check )
    {
        if ( check > 1 ) {
            x0 = malloc( size );
        }

        spmGenRHS( SpmRhsI, nrhs, spm, x0, spm->n, b, spm->n );
        memcpy( x, b, size );
    }
//    else {
//        spmGenRHS( SpmRhsRndB, nrhs, spm, NULL, spm->n, x, spm->n );

        // Apply also normalization to b vectors /
        spmScalVector( spm->flttype, 1./normA, spm->n * nrhs, b, 1 );

        // Save b for refinement /
        memcpy( b, x, size );
    }
*/

//for(int i = 0; i < spm->n; i++){
//      printf ("%lf\n", ((double*)b)[i]);      
//}
    /**
     * Solve the linear system (and perform the optional refinement)
     */
    pastix_task_solve( pastix_data, nrhs, x, spm->n );
 //   pastix_task_refine( pastix_data, spm->n, nrhs, b, spm->n, x, spm->n );

    if ( check )
    {
        rc = spmCheckAxb( dparm[DPARM_EPSILON_REFINEMENT], nrhs, spm, x0, spm->n, b, spm->n, x, spm->n );

        if ( x0 ) {
            free( x0 );
        }
    }
        FILE *f = fopen("X.sol", "w");
        for (int i = 0; i < spm->n; ++i) {
            fprintf(f,"%.17f\n", ((double*)x)[i]);
        }
        fclose(f);

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
