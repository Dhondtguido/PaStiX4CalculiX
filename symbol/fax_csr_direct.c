/**
 *
 * @file fax_csr_direct.c
 *
 * This file contains routines to create the graph of the direct
 * factorization of a graph A.
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
#include "pastix/order.h"
#include "fax_csr.h"

/**
 *******************************************************************************
 *
 * @ingroup symbol_fax_dev
 *
 * @brief Compute the non zero pattern of the direct factorization of a matrix
 * A, given the supernode partition associated.
 *
 *******************************************************************************
 *
 * @param[in] graphA
 *          The graph structure of the original matrix A to factorize.
 *
 * @param[in] order
 *          The order structure holding the number of cblk and the partition
 *
 * @param[out] graphL
 *          The graph the structure of the non zero pattern of the factorized
 *          matrix.  On entry, a pointer to a graph structure. No need for
 *          initialization.  On exit, the structure contains the computed graph.
 *
 *******************************************************************************
 *
 * @retval >=0, the number of non zero entries in the generated graph.
 * @retval -i, if the i^th parameter is incorrect
 *
 *******************************************************************************/
pastix_int_t
faxCSRFactDirect( const fax_csr_t *graphA, const pastix_order_t *order, fax_csr_t *graphL )
{
    pastix_int_t        i, k, nnz;
    pastix_int_t        nnznbr, father;
    pastix_int_t       *tmp     = NULL;
    pastix_int_t       *ja      = NULL;
    const pastix_int_t  cblknbr = order->cblknbr;
    const pastix_int_t *rangtab = order->rangtab;
    const pastix_int_t *treetab = order->treetab;

    /* Check parameters */
    if ( graphA == NULL ) {
        return -1;
    }
    if ( order == NULL ) {
        return -2;
    }
    if ( graphL == NULL ) {
        return -3;
    }

    if ( ( order->cblknbr < 0 ) || ( order->cblknbr > graphA->n ) ) {
        return -4;
    }

    /* Quick return */
    if ( graphA->n == 0 ) {
        return 0;
    }

    MALLOC_INTERN( tmp, graphA->n, pastix_int_t );

    /* Compute the nnz structure of each supernode in A */
    faxCSRCblkCompress( graphA, order, graphL, tmp );

    /* Compute the symbolic factorization */
    for ( k = 0; k < cblknbr; k++, treetab++ ) {
        father = *treetab;

        /* Merge son's nodes into father's list */
        if ( ( father != k ) && ( father > 0 ) ) {
            i  = 0;
            ja = graphL->rows[k];

            /* Take only the trows outside the cblk */
            while ( ( i < graphL->nnz[k] ) && ( ja[i] < rangtab[k+1] ) ) {
                i++;
            }

            nnznbr = pastix_intset_union( graphL->nnz[k] - i,
                                          graphL->rows[k] + i,
                                          graphL->nnz[father],
                                          graphL->rows[father],
                                          tmp );

            memFree( graphL->rows[father] );
            MALLOC_INTERN( graphL->rows[father], nnznbr, pastix_int_t );
            memcpy( graphL->rows[father], tmp, sizeof( pastix_int_t ) * nnznbr );
            graphL->nnz[father] = nnznbr;
        }
    }

#if defined( PASTIX_DEBUG_SYMBOL )
    /* Check that all terms of A are in the pattern */
    {
        pastix_int_t ind;
        for ( k = 0; k < cblknbr; k++ ) {
            /* Put the diagonal elements (A does not contains them) */
            for ( i = rangtab[k]; i < rangtab[k + 1]; i++ ) {
                j = 0;
                while ( ( j < graphA->nnz[i] ) && ( graphA->rows[i][j] < i ) ) {
                    j++;
                }

                for ( ind = j; ind < graphA->nnz[i]; ind++ ) {
                    assert( graphA->rows[i][ind] >= i );
                }
                for ( ind = j + 1; ind < graphA->nnz[i]; ind++ ) {
                    assert( graphA->rows[i][ind] > graphA->rows[i][ind - 1] );
                }

                ind = pastix_intset_union(
                    graphL->nnz[k], graphL->rows[k], graphA->nnz[i] - j, graphA->rows[i] + j, tmp );

                assert( ind <= graphL->nnz[k] );
            }
        }
    }
#endif

    memFree( tmp );

    /*
     * Computes the nnz in graphL
     */
    nnz = 0;
    for ( i = 0; i < cblknbr; i++ ) {
        pastix_int_t ncol, nrow;
        ncol = rangtab[i + 1] - rangtab[i];
        nrow = graphL->nnz[i];

        assert( nrow >= ncol );
        assert( nrow <= graphA->n );

        nnz += ( ncol * ( ncol + 1 ) ) / 2;
        nnz += ( ncol * ( nrow - ncol ) );
    }

    graphL->total_nnz = nnz;
    return nnz;
}
