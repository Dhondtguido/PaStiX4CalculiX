/**
 *
 * @file order_amalgamate.c
 *
 * PaStiX amalgamation main routine to apply after ordering strategies that do
 * not provide supernodes.
 *
 * @copyright 2004-2018 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @version 6.0.1
 * @author Pascal Henon
 * @author Mathieu Faverge
 * @date 2018-07-16
 *
 **/
#include "common.h"
#include "graph.h"
#include "pastix/order.h"
#include "fax_csr.h"

/**
 *******************************************************************************
 *
 * @ingroup pastix_order

 * @brief Update the order structure with an amalgamation algorithm.
 *
 * This algorithm almagamates small blocks together in larger ones, and generates
 * an updated order structure. See after for the different parameters.
 *
 *******************************************************************************
 *
 * @param[in] verbose
 *          Adjust the level of verbosity of the function
 *
 * @param[in] ilu
 *          - 1: incomplete factorization will be performed.
 *          - 0: direct factorization will be performed.
 *
 * @param[in] levelk
 *          Unused if ilu == 0.
 *          - k >= 0: ordering for ILU(k) factorization will be generated.
 *          - < 0: ordering for direct factorization will be generated.
 *
 * @param[in] rat_cblk
 *          Must be >= 0. Fill ratio that limits the amalgamation process based
 *          on the graph structure.
 *
 * @param[in] rat_blas
 *          Must be >= rat_cblk. Fill ratio that limits the amalgamation process
 *          that merges blocks in order to reduce the BLAS computational time
 *          (see amalgamate() for further informations).
 *
 * @param[inout] symbmtx
 *          The symbol matrix structure to construct. On entry, the initialized
 *          structure (see pastixSymbolInit()). On exit, the symbol matrix generated
 *          after the amalgamation process.
 *
 * @param[inout] csc
 *          The original csc for which the symbol matrix needs to be generated.
 *          Rebase to C numbering on exit.
 *
 * @param[inout] orderptr
 *          The oder structure that contains the perm and invp array generated
 *          by the ordering step. The supernode partition might be initialized
 *          or not.
 *          On exit, it is rebased to c numbering and contains the updated
 *          perm/invp arrays as well as the supernode partition.
 *
 * @param[in] pastix_comm
 *          The PaStiX instance communicator.
 *
 *******************************************************************************
 *
 * @retval PASTIX_SUCCESS on success.
 * @retval PASTIX_ERR_ALLOC if allocation went wrong.
 * @retval PASTIX_ERR_BADPARAMETER if incorrect parameters are given.
 *
 *******************************************************************************/
int
pastixOrderAmalgamate( int             verbose,
                       int             ilu,
                       int             levelk,
                       int             rat_cblk,
                       int             rat_blas,
                       pastix_graph_t *csc,
                       pastix_order_t *orderptr,
                       MPI_Comm        pastix_comm )
{
    fax_csr_t     graphPA, graphL;
    pastix_int_t  n;
    pastix_int_t  nnzA, nnzL;
    Clock         timer;
    int           procnum;

    MPI_Comm_rank( pastix_comm, &procnum );

    /* Check parameters correctness */
    if ( ( ilu == 0 ) || ( levelk < 0 ) ) {
        /* Forces levelk to -1 */
        levelk = -1;
    }
    if ( csc == NULL ) {
        errorPrintW( "pastixOrderAmalgamate: wrong parameter csc" );
        return PASTIX_ERR_BADPARAMETER;
    }
    if ( orderptr == NULL ) {
        errorPrintW( "pastixOrderAmalgamate: wrong parameter orderptr" );
        return PASTIX_ERR_BADPARAMETER;
    }

    /* Convert Fortran to C numbering if not already done */
    graphBase( csc, 0 );
    pastixOrderBase( orderptr, 0 );

    n = csc->n;

    /* Create the graph of P A */
    faxCSRGenPA( csc, orderptr->permtab, &graphPA );

    if ( verbose > PastixVerboseYes ) {
        pastix_print( procnum,
                      0,
                      "Level of fill = %ld\n"
                      "Amalgamation ratio: cblk = %d, blas = %d\n",
                      (long)levelk,
                      rat_cblk,
                      rat_blas );
    }

    /*
     * Compute the graph of the factorized matrix L before amalgamation
     * and the associated treetab and nodetab
     */
    /* Direct Factorization */
    if ( ( ilu == 0 ) || ( levelk == -1 ) )
    {
        clockStart( timer );
        nnzL = faxCSRFactDirect( &graphPA, orderptr, &graphL );
        clockStop( timer );

        if ( verbose > PastixVerboseYes ) {
            pastix_print( procnum,
                          0,
                          "Time to compute scalar symbolic direct factorization  %.3g s\n",
                          clockVal( timer ) );
        }
    }
    /* ILU(k) Factorization */
    else
    {
        clockStart( timer );
        nnzL = faxCSRFactILUk( &graphPA, orderptr, levelk, &graphL );
        clockStop( timer );

        if ( verbose > PastixVerboseYes ) {
            pastix_print( procnum,
                          0,
                          "Time to compute scalar symbolic factorization of ILU(%ld) %.3g s\n",
                          (long)levelk,
                          clockVal( timer ) );
        }
    }

    nnzA = ( faxCSRGetNNZ( &graphPA ) + n ) / 2;
    faxCSRClean( &graphPA );

    if ( verbose > PastixVerboseYes ) {
        pastix_print( procnum,
                      0,
                      "Scalar nnza = %ld nnzlk = %ld, fillrate0 = %.3g \n",
                      (long)nnzA,
                      (long)nnzL,
                      (double)nnzL / (double)nnzA );
    }

    /*
     * Amalgamate the blocks
     */
    clockStart( timer );
    faxCSRAmalgamate( ilu,
                      (double)rat_cblk / 100.,
                      (double)rat_blas / 100.,
                      &graphL,
                      orderptr,
                      pastix_comm );
    clockStop( timer );

    faxCSRClean( &graphL );

    if ( verbose > PastixVerboseYes ) {
        pastix_print( procnum,
                      0,
                      "Time to compute the amalgamation of supernodes %.3g s\n",
                      clockVal( timer ) );
        pastix_print( procnum,
                      0,
                      "Number of cblk in the amalgamated symbol matrix = %ld \n",
                      (long)orderptr->cblknbr );
    }

    return PASTIX_SUCCESS;
}
