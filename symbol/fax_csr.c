/**
 *
 * @file fax_csr.c
 *
 * PaStiX Fax csr routines to handle the graphs of PA and L
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
 * @ingroup symbol_fax_dev
 *
 * @brief Initialize the data structure by doing the first allocations within
 * the structure and initializing the fields.
 *
 *******************************************************************************
 *
 * @param[in] n
 *          The size of the graph that needs to be initialized.
 *
 * @param[out] csr
 *          The graph to initialize.
 *
 *******************************************************************************/
void
faxCSRInit( pastix_int_t n, fax_csr_t *csr )
{
    csr->n         = n;
    csr->total_nnz = 0;
    MALLOC_INTERN( csr->nnz, n, pastix_int_t );
    MALLOC_INTERN( csr->rows, n, pastix_int_t * );

    memset( csr->nnz, 0, n * sizeof( pastix_int_t ) );
    memset( csr->rows, 0, n * sizeof( pastix_int_t * ) );
}

/**
 *******************************************************************************
 *
 * @ingroup symbol_fax_dev
 *
 * @brief Free the data store in the structure.
 *
 *******************************************************************************
 *
 * @param[inout] csr
 *          The graph to clean.
 *
 *******************************************************************************/
void
faxCSRClean( fax_csr_t *csr )
{
    pastix_int_t i;
    for ( i = 0; i < csr->n; i++ ) {
        if ( csr->nnz[i] != 0 ) {
            memFree_null( csr->rows[i] );
        }
    }
    memFree_null( csr->rows );
    memFree_null( csr->nnz );
}

/**
 *******************************************************************************
 *
 * @ingroup symbol_fax_dev
 *
 * @brief Computes the number of non zero entries in the graph.
 *
 * It is using the following formula: nnz = sum( i=0..n, nnz[n] )
 * The formula must be post computed to adapt to presence of diagonal elements
 * or not, and to the symmetry of the graph.
 *
 *******************************************************************************
 *
 * @param[in] csr
 *          The graph on which the number of non zero entries is computed.
 *
 *******************************************************************************
 *
 * @retval The number of non zero entries.
 *
 *******************************************************************************/
pastix_int_t
faxCSRGetNNZ( const fax_csr_t *csr )
{
    pastix_int_t i, nnz;
    nnz = 0;
    for ( i = 0; i < csr->n; i++ ) {
        nnz += csr->nnz[i];
    }
    return nnz;
}

/**
 *******************************************************************************
 *
 * @ingroup symbol_fax_dev
 *
 * @brief Compact a compressed graph.
 *
 * All nodes with no non zero entries are removed from the graph, the allocated
 * space is not adjusted.
 *
 *******************************************************************************
 *
 * @param[inout] csr
 *          The graph to compact.
 *          On entry, graph which might contain nodes with no non zero entries.
 *          On exit, all those nodes are suppressed and the compressed graph is
 *          returned.
 *
 *******************************************************************************/
void
faxCSRCompact( fax_csr_t *csr )
{
    pastix_int_t n = csr->n;
    pastix_int_t i, j;

    /* Look for first deleted node */
    for ( i = 0, j = 0; i < n; i++, j++ ) {
        if ( csr->nnz[i] == 0 )
            break;
    }

    /* Compact the nodes */
    for ( ; i < n; i++ ) {
        if ( csr->nnz[i] > 0 ) {
            assert( j < i );
            csr->nnz[j]  = csr->nnz[i];
            csr->rows[j] = csr->rows[i];
            csr->nnz[i]  = 0;
            csr->rows[i] = NULL;
            j++;
        }
    }

    csr->n = j;
}

/**
 *******************************************************************************
 *
 * @ingroup symbol_fax_dev
 *
 * @brief Generate the graph of P*A from the graph of A and the
 * permutation vector.
 *
 *******************************************************************************
 *
 * @param[in] graphA
 *          The original graph Aon which the permutation will be applied.
 *
 * @param[in] perm
 *          Integer array of size graphA->n. Contains the permutation to apply to A.
 *
 * @param[inout] graphPA
 *          On entry, the initialized graph with size graphA->n.
 *          On exit, contains the graph of P A.
 *
 *******************************************************************************
 *
 * @retval PASTIX_SUCCESS on success.
 * @retval PASTIX_ERR_ALLOC if allocation went wrong.
 * @retval PASTIX_ERR_BADPARAMETER if incorrect parameters are given.
 *
 *******************************************************************************/
int
faxCSRGenPA( const pastix_graph_t *graphA, const pastix_int_t *perm, fax_csr_t *graphPA )
{
    pastix_int_t *rowsPA, *rowsA;
    pastix_int_t  i, j, ip;
    pastix_int_t  baseval;
    pastix_int_t  n = graphPA->n = graphA->n;

    baseval = graphA->colptr[0];

    faxCSRInit( graphA->n, graphPA );

    /* Compute the number of nnz per vertex */
    for ( i = 0; i < n; i++ ) {
        ip = perm[i];
        /* Add diagonal (could be removed fro direct) */
        graphPA->nnz[ip] = graphA->colptr[i + 1] - graphA->colptr[i] + 1;
    }

    /* Create the row vector */
    for ( i = 0; i < n; i++ ) {
        ip = perm[i] - baseval;

        MALLOC_INTERN( graphPA->rows[ip], graphPA->nnz[ip], pastix_int_t );

        rowsPA = graphPA->rows[ip];
        rowsA  = graphA->rows + graphA->colptr[i] - baseval;

        /* Add diagonal */
        *rowsPA = ip;
        rowsPA++;

        for ( j = 1; j < graphPA->nnz[ip]; j++, rowsPA++ ) {
            *rowsPA = perm[*rowsA];
            rowsA++;
        }

        intSort1asc1( graphPA->rows[ip], graphPA->nnz[ip] );
    }
    return PASTIX_SUCCESS;
}

/**
 *******************************************************************************
 *
 * @ingroup symbol_fax_dev
 *
 * @brief Compact a element wise graph of a matrix A, according to the
 * associated partition.
 *
 *******************************************************************************
 *
 * @param[in] graphA
 *          The original graph of A element wise to compress.
 *
 * @param[in] order
 *          The ordering structure that holds the partition associated to graphAo.
 *
 * @param[out] graphL
 *          On entry, the block wise initialized graph of A with size order->cblknbr.
 *          On exit, contains the compressed graph of A.
 *
 * @param[inout] work
 *          Workspace of size >= max( degree(L_i) ), so >= grapA->n.
 *
 *******************************************************************************/
void
faxCSRCblkCompress( const fax_csr_t      *graphA,
                    const pastix_order_t *order,
                    fax_csr_t            *graphL,
                    pastix_int_t         *work )
{
    pastix_int_t        i, j, k, nnznbr;
    const pastix_int_t  cblknbr = order->cblknbr;
    const pastix_int_t *rangtab = order->rangtab;
    pastix_int_t       *work2;
    pastix_int_t       *tmp, *tmp1, *tmp2;

    MALLOC_INTERN( work2, graphA->n, pastix_int_t );
    tmp1 = work;
    tmp2 = work2;

    assert( order->baseval == 0 );
    faxCSRInit( cblknbr, graphL );

    /*
     * Let's accumulate the row presents in the column blok k in tmp1
     * Then use tmp2 to merge the elements of the next column, and tmp to switch
     * pointers.
     */
    for ( k = 0; k < cblknbr; k++ ) {
        /* Put the diagonal elements (In case A does not contains them) */
        j = 0;
        for ( i = rangtab[k]; i < rangtab[k + 1]; i++ ) {
            tmp1[j++] = i;
        }
        nnznbr = j;

        for ( i = rangtab[k]; i < rangtab[k + 1]; i++ ) {
            j = 0;

            /* We take only the elements greater than i */
            while ( ( j < graphA->nnz[i] ) && ( graphA->rows[i][j] <= i ) ) {
                j++;
            }

            /* Merge the actual list with the edges of the ith vertex */
            nnznbr =
                pastix_intset_union( nnznbr, tmp1, graphA->nnz[i] - j, graphA->rows[i] + j, tmp2 );

            /* Swap tmp1 and the merged set tmp */
            tmp  = tmp1;
            tmp1 = tmp2;
            tmp2 = tmp;
        }

#if !defined( NDEBUG ) && defined( PASTIX_DEBUG_SYMBOL )
        /* Check that the first elements are the diagonal ones */
        {
            pastix_int_t ind;
            ind = 0;
            assert( nnznbr >= ( rangtab[k + 1] - rangtab[k] ) );
            for ( j = rangtab[k]; j < rangtab[k + 1]; j++ ) {
                assert( tmp1[ind] == j );
                ind++;
            }
            assert( nnznbr > 0 );
        }
#endif

        /* Update graphL */
        graphL->nnz[k] = nnznbr;
        MALLOC_INTERN( graphL->rows[k], nnznbr, pastix_int_t );
        memcpy( graphL->rows[k], tmp1, sizeof( pastix_int_t ) * nnznbr );
    }

    memFree( work2 );
}
