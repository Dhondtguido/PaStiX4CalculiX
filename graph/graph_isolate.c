/**
 *
 * @file graph_isolate.c
 *
 * PaStiX graph isolate routine
 *
 * @copyright 2004-2018 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @version 6.0.1
 * @author Xavier Lacoste
 * @author Pierre Ramet
 * @author Mathieu Faverge
 * @date 2018-07-16
 *
 **/
#include "common.h"
#include "pastix/order.h"
#include "blend/extendVector.h"
#include "graph.h"

/**
 *******************************************************************************
 *
 * @ingroup pastix_graph
 *
 * @brief Isolate a subset of vertices from a given graph.
 *
 * Return a new graph cleaned from those vertices.
 *
 *******************************************************************************
 *
 * @param[in] n
 *          The number of columns of the original GRAPH matrix.
 *
 * @param[in] colptr
 *          Array of size n+1.
 *          Index of first element of each column in rows array.
 *
 * @param[in] rows
 *          Array of size nnz = colptr[n] - colptr[0].
 *          Rows of each non zero entries.
 *
 * @param[in] isolate_n
 *          The number of columns to isolate from the original graph.
 *
 * @param[inout] isolate_list
 *          Array of size isolate_n.
 *          List of columns to isolate. On exit, the list is sorted by ascending
 *          indexes. Must be based as the graph.
 *
 * @param[out] new_colptr
 *          Array of size n-isolate_n+1.
 *          Index of first element of each column in rows array for the new graph.
 *          If new_colptr == NULL, nothing is returned, otherwise the pointer to
 *          the allocated structure based as the input colptr.
 *
 * @param[out] new_rows
 *          Array of size new_nnz = (*new_colptr)[n] - (*new_colptr)[0].
 *          Rows of each non zero entries for the new graph.
 *          If new_rows == NULL, nothing is returned, otherwise the pointer to
 *          the allocated structure based as the input rows.
 *
 * @param[out] new_perm
 *          Array of size n-isolate_n.
 *          Contains permutation generated to isolate the columns at the end of
 *          the graph that is 0-based.
 *          If new_perm == NULL, nothing is returned, otherwise the pointer to
 *          the allocated structure.
 *
 * @param[out] new_invp
 *          Array of size n-isolate_n.
 *          Contains the inverse permutation generated to isolate the columns
 *          at the end of the graph that is 0-based.
 *          If new_invp == NULL, nothing is returned, otherwise the pointer to
 *          the allocated structure.
 *
 *******************************************************************************
 *
 * @retval PASTIX_SUCCESS on success,
 * @retval PASTIX_ERR_ALLOC if allocation went wrong,
 * @retval PASTIX_ERR_BADPARAMETER if incorrect parameters are given.
 *
 *******************************************************************************/
int graphIsolate(       pastix_int_t   n,
                  const pastix_int_t  *colptr,
                  const pastix_int_t  *rows,
                        pastix_int_t   isolate_n,
                        pastix_int_t  *isolate_list,
                        pastix_int_t **new_colptr,
                        pastix_int_t **new_rows,
                        pastix_int_t **new_perm,
                        pastix_int_t **new_invp )
{
    pastix_int_t *tmpcolptr = NULL;
    pastix_int_t *tmprows   = NULL;
    pastix_int_t *tmpperm   = NULL;
    pastix_int_t *tmpinvp   = NULL;
    pastix_int_t  baseval = colptr[0];
    pastix_int_t  nnz = colptr[n] - baseval;
    pastix_int_t  new_n = n - isolate_n;
    pastix_int_t  new_nnz;
    pastix_int_t  i, j, ip, k;
    pastix_int_t  iter_isolate = 0;
    pastix_int_t  iter_non_isolate  = 0;

    if ( (isolate_n > n)  || (isolate_n < 0) ) {
        errorPrintW("Number of columns to isolate greater than the columns in the GRAPH matrix\n");
        return PASTIX_ERR_BADPARAMETER;
    }

    /* Quick Return */
    if (isolate_n == 0) {
        if (new_colptr != NULL) *new_colptr = (pastix_int_t*)colptr;
        if (new_rows   != NULL) *new_rows   = (pastix_int_t*)rows;
        return PASTIX_SUCCESS;
    }

    if (isolate_n == n) {
        if (new_colptr != NULL) {
            MALLOC_INTERN(*new_colptr, n, pastix_int_t);
            memcpy( *new_colptr, colptr, n*sizeof(pastix_int_t) );
        }
        if (new_rows != NULL) {
            MALLOC_INTERN(*new_rows, nnz, pastix_int_t);
            memcpy( *new_rows, rows, nnz*sizeof(pastix_int_t) );
        }
        if (new_perm != NULL) {
            MALLOC_INTERN(*new_perm, n, pastix_int_t);
            for (i = 0; i < n; i++) {
                (*new_perm)[i] = i;
            }
        }
        if (new_invp != NULL) {
            MALLOC_INTERN(*new_invp, n, pastix_int_t);
            for (i = 0; i < n; i++) {
                (*new_invp)[i] = i;
            }
        }
        return PASTIX_SUCCESS;
    }

    /* Sort the lost of vertices */
    intSort1asc1(isolate_list, isolate_n);

    /* Init invp array */
    MALLOC_INTERN(tmpinvp, n, pastix_int_t);
    for (i = 0; i <n; i++) {
        if ((iter_isolate < isolate_n) &&
            (i == isolate_list[iter_isolate]-baseval))
        {
            tmpinvp[new_n+iter_isolate] = i;
            iter_isolate++;
        }
        else
        {
            tmpinvp[iter_non_isolate] = i;
            iter_non_isolate++;
        }
    }

    assert(iter_non_isolate == new_n    );
    assert(iter_isolate     == isolate_n);

    /* Init perm array */
    MALLOC_INTERN(tmpperm, n, pastix_int_t);
    for(i = 0; i < n; i++)
        tmpperm[tmpinvp[i]] = i;

#if defined(PASTIX_DEBUG_GRAPH)
    for(i = 0; i < n; i++)
    {
        assert(tmpperm[i] < n );
        assert(tmpperm[i] > -1);
    }
#endif

    /* Create the new_colptr array */
    MALLOC_INTERN(tmpcolptr, new_n + 1, pastix_int_t);
    memset(tmpcolptr, 0, (new_n + 1)*sizeof(pastix_int_t));

    tmpcolptr[0] = baseval;
    for (i=0; i<n; i++)
    {
        ip = tmpperm[i];
        if (ip < new_n)
        {
            for (j = colptr[i]-baseval; j < colptr[i+1]-baseval; j++)
            {
                /* Count edges in each column of the new graph */
                if (tmpperm[rows[j]-baseval] < new_n)
                {
                    tmpcolptr[ip+1]++;
                }
            }
        }
    }

    for (i = 0; i < new_n; i++)
        tmpcolptr[i+1] += tmpcolptr[i];

    new_nnz = tmpcolptr[new_n] - tmpcolptr[0];

    /* Create the new rows array */
    if ( new_nnz != 0 ) {
        MALLOC_INTERN(tmprows, new_nnz, pastix_int_t);
        for (i = 0; i <n; i++)
        {
            ip = tmpperm[i];
            if (ip < new_n)
            {
                k = tmpcolptr[ip]-baseval;
                for (j = colptr[i]-baseval; j < colptr[i+1]-baseval; j ++)
                {
                    /* Count edges in each column of the new graph */
                    if (tmpperm[rows[j]-baseval] < new_n)
                    {
                        tmprows[k] = tmpperm[rows[j]-baseval] + baseval;
                        k++;
                    }
                }
                assert( k == tmpcolptr[ip+1]-baseval );
            }
        }
    }

    if (new_colptr != NULL) {
        *new_colptr = tmpcolptr;
    } else {
        memFree_null( tmpcolptr );
    }
    if (new_rows != NULL) {
        *new_rows = tmprows;
    } else {
        memFree_null( tmprows );
    }
    if (new_perm != NULL) {
        *new_perm = tmpperm;
    } else {
        memFree_null( tmpperm );
    }
    if (new_invp != NULL) {
        *new_invp = tmpinvp;
    } else {
        memFree_null( tmpinvp );
    }

    return PASTIX_SUCCESS;
}

/**
 *******************************************************************************
 *
 * @ingroup pastix_graph
 *
 * @brief Isolate the subgraph associated to a range of unknowns in the permuted
 * graph.
 *
 * This routine isolates a continuous subset of vertices from a given graph, and
 * returns a new graph made of those vertices and internal connexions. Extra
 * edges are created between vertices if they are connected through a halo at a
 * distance d given in parameter.
 *
 *******************************************************************************
 *
 * @param[in] graph
 *          The original graph associated from which vertices and edges must be
 *          extracted.
 *
 * @param[in] order
 *          The ordering structure associated to the graph.
 *
 * @param[inout] out_graph
 *          The extracted graph. If the graph is allocated, it is freed before usage.
 *          On exit, contains the subgraph of the vertices invp[fnode] to invp[lnode-1].
 *
 * @param[in] fnode
 *          The index of the first node to extract in the inverse permutation.
 *
 * @param[in] lnode
 *          The index (+1) of the last node to extract in the inverse permutation.
 *
 * @param[in] distance
 *          Distance considered in number of edges to create an edge in isolated
 *          graph.
 *
 *******************************************************************************
 *
 * @retval PASTIX_SUCCESS on success.
 * @retval PASTIX_ERR_ALLOC if allocation went wrong.
 * @retval PASTIX_ERR_BADPARAMETER if incorrect parameters are given.
 *
 *******************************************************************************/
int
graphIsolateRange( const pastix_graph_t *graph,
                   const pastix_order_t *order,
                         pastix_graph_t *out_graph,
                         pastix_int_t    fnode,
                         pastix_int_t    lnode,
                         pastix_int_t    distance )
{
    ExtendVectorINT     vec;
    pastix_int_t        baseval = graph->colptr[0];
    pastix_int_t        n       = graph->n;;
    const pastix_int_t *colptr  = graph->colptr;
    const pastix_int_t *rows    = graph->rows;
    const pastix_int_t *perm    = order->permtab;
    const pastix_int_t *invp    = order->peritab;
    pastix_int_t  out_n = lnode - fnode;
    pastix_int_t  out_nnz;
    pastix_int_t *out_colptr;
    pastix_int_t *out_rows;
    pastix_int_t  k, i, ip, jj, j, jp, sze, d;
    pastix_int_t *out_connected;
    pastix_int_t  row_counter;
    int ret = PASTIX_SUCCESS;

    if ( out_graph == NULL ) {
        errorPrintW( "graphIsolateSupernode: Incorrect pointer for the output graph\n");
        return PASTIX_ERR_BADPARAMETER;
    }

    n             = graph->n;
    out_graph->n  = out_n;
    out_graph->gN = out_n;

    if ( out_n == 0 ) {
        errorPrintW( "graphIsolateSupernode: Empty supernode\n");
        return PASTIX_ERR_BADPARAMETER;
    }

    /* Quick Return */
    if ( out_n == n ) {
        /* Only one supernode */
        assert( order->cblknbr == 1 );
        graphCopy( out_graph, graph );
        return PASTIX_SUCCESS;
    }

    /* Create the new_colptr array */
    MALLOC_INTERN( out_graph->colptr, out_n + 1, pastix_int_t );
    memset( out_graph->colptr, 0, (out_n + 1) * sizeof(pastix_int_t) );
    out_colptr = out_graph->colptr;

    /* Temporary array of connections to avoid double counting when extending */
    MALLOC_INTERN(out_connected, out_n, pastix_int_t);

    /* (i,j) in permutated ordering */
    /* (ip,jp) in initial ordering */
    out_colptr[0] = baseval;

    extendint_Init( &vec, 100 );

    /*
     * The first loop counts the number of edges
     */
    for (ip=fnode; ip<lnode; ip++)
    {
        extendint_Clear( &vec );
        memset(out_connected, 0, (out_n) * sizeof(pastix_int_t));
        out_connected[ip-fnode] = 1;

        /* i^th vertex in the initial numbering */
        extendint_Add( &vec, invp[ip] );
        sze =  1;
        d   = -1;
        k   =  0;

        while( d < distance ) {
            for(; k<sze; k++) {
                i = extendint_Read( &vec, k );

                for (jj = colptr[i  ]-baseval;
                     jj < colptr[i+1]-baseval; jj++) {

                    j  = rows[jj]-baseval;
                    jp = perm[j];

                    /* Count edges in each column of the new graph */
                    if ( ( jp >= fnode ) && ( jp < lnode ) ) {
                        if (out_connected[jp-fnode] == 0){
                            out_connected[jp-fnode] = 1;
                            out_colptr[ip-fnode+1]++;
                        }
                    }
                    else {
                        extendint_Add( &vec, j );
                    }
                }
            }
            d++;
            sze = extendint_Size( &vec );
        }
    }

    /* Update the colptr */
    for (i = 0; i < out_n; i++){
        out_colptr[i+1] += out_colptr[i];
    }

    out_nnz = out_colptr[out_n] - out_colptr[0];

    /* Allocation will fail if matrix is diagonal and no off-diagonal elements are found */
    if ( out_nnz == 0 ){
        fprintf( stderr, "Diagonal matrix cannot be correcly managed here!\n" );
        //return EXIT_FAILURE;
    }

    /* Create the new rows array */
    MALLOC_INTERN( out_graph->rows, out_nnz, pastix_int_t );
    out_rows = out_graph->rows;
    row_counter = 0;

    /*
     * The second loop initialize the row array
     */
    for (ip=fnode; ip<lnode; ip++){
        extendint_Clear( &vec );
        memset(out_connected, 0, out_n * sizeof(pastix_int_t));
        out_connected[ip-fnode] = 1;

        /* i^th vertex in the initial numbering */
        extendint_Add( &vec, invp[ip] );
        sze =  1;
        d   = -1;
        k   =  0;

        while( d < distance ) {
            for(; k<sze; k++) {
                i = extendint_Read( &vec, k );

                for (jj = colptr[i  ]-baseval;
                     jj < colptr[i+1]-baseval; jj++) {

                    j  = rows[jj]-baseval;
                    jp = perm[j];

                    /* Count edges in each column of the new graph */
                    if ( ( jp >= fnode ) && ( jp < lnode ) )
                    {
                        if (out_connected[jp-fnode] == 0){
                            out_connected[jp-fnode] = 1;
                            out_rows[row_counter] = jp-fnode;
                            row_counter++;
                        }
                    }
                    else {
                        extendint_Add( &vec, j );
                    }
                }
            }
            d++;
            sze = extendint_Size( &vec );
        }
    }

    extendint_Exit( &vec );
    free(out_connected);
    graphBase( out_graph, 0 );

    return ret;
}
