/**
 *
 * @file graph.c
 *
 * PaStiX graph structure routines
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
 * @addtogroup pastix_graph
 * @{
 *
 **/
#include "common.h"
#include "graph.h"

/**
 *******************************************************************************
 *
 * @brief Free the content of the graph structure.
 *
 *******************************************************************************
 *
 * @param[inout] graph
 *          The pointer graph structure to free.
 *
 *******************************************************************************/
void graphExit( pastix_graph_t *graph )
{
    /* Parameter checks */
    if ( graph == NULL ) {
        errorPrint("graphClean: graph pointer is NULL");
        return;
    }

    graph->gN = 0;
    graph->n  = 0;

    if ( graph->colptr != NULL) {
        memFree_null(graph->colptr);
    }

    if (graph->rows != NULL) {
        memFree_null(graph->rows);
    }

    if (graph->loc2glob != NULL) {
        memFree_null( graph->loc2glob );
    }

    return;
}

/**
 *******************************************************************************
 *
 * @brief Rebase the graph to the given value.
 *
 *******************************************************************************
 *
 * @param[inout] graph
 *          The graph to rebase.
 *
 * @param[in] baseval
 *          The base value to use in the graph (0 or 1).
 *
 *******************************************************************************/
void graphBase( pastix_graph_t *graph,
                int             baseval )
{
    pastix_int_t baseadj;
    pastix_int_t i, n, nnz;

    /* Parameter checks */
    if ( graph == NULL ) {
        errorPrint("graphBase: graph pointer is NULL");
        return;
    }
    if ( (graph->colptr == NULL) ||
         (graph->rows   == NULL) )
    {
        errorPrint("graphBase: graph pointer is not correctly initialized");
        return;
    }
    if ( (baseval != 0) &&
         (baseval != 1) )
    {
        errorPrint("graphBase: baseval is incorrect, must be 0 or 1");
        return;
    }

    baseadj = baseval - graph->colptr[0];
    if (baseadj == 0)
	return;

    n   = graph->n;
    nnz = graph->colptr[n] - graph->colptr[0];

    for (i = 0; i <= n; i++) {
        graph->colptr[i]   += baseadj;
    }
    for (i = 0; i < nnz; i++) {
        graph->rows[i] += baseadj;
    }

    if (graph->loc2glob != NULL) {
        for (i = 0; i < n; i++) {
            graph->loc2glob[i] += baseadj;
        }
    }
    return;
}

/**
 *******************************************************************************
 *
 * @ingroup pastix_graph
 *
 * @brief This routine copy a given ordering in a new one.
 *
 * This function copies a graph structure into another one. If all subpointers
 * are NULL, then they are all allocated and contains the original graphsrc
 * values on exit. If one or more array pointers are not NULL, then, only those
 * are copied to the graphdst structure.
 *
 *******************************************************************************
 *
 * @param[inout] graphdst
 *          The destination graph
 *
 * @param[in] graphsrc
 *          The source graph
 *
 *******************************************************************************
 *
 * @retval PASTIX_SUCCESS on successful exit
 * @retval PASTIX_ERR_BADPARAMETER if one parameter is incorrect.
 *
 *******************************************************************************/
int
graphCopy( pastix_graph_t       *graphdst,
           const pastix_graph_t *graphsrc )
{
    /* Parameter checks */
    if ( graphdst == NULL ) {
        return PASTIX_ERR_BADPARAMETER;
    }
    if ( graphsrc == NULL ) {
        return PASTIX_ERR_BADPARAMETER;
    }
    if ( graphsrc == graphdst ) {
        return PASTIX_ERR_BADPARAMETER;
    }

    graphdst->gN  = graphsrc->gN;
    graphdst->n   = graphsrc->n;
    graphdst->dof = graphsrc->dof;
    graphdst->colptr   = NULL;
    graphdst->rows     = NULL;
    graphdst->loc2glob = NULL;
    graphdst->glob2loc = NULL;

    if ( graphsrc->colptr != NULL )
    {
        MALLOC_INTERN( graphdst->colptr, graphsrc->n + 1, pastix_int_t );
        memcpy( graphdst->colptr, graphsrc->colptr, (graphsrc->n+1) * sizeof(pastix_int_t) );
    }

    if ( graphsrc->rows != NULL )
    {
        pastix_int_t nnz = graphdst->colptr[graphdst->n] - graphdst->colptr[0];
        MALLOC_INTERN( graphdst->rows, nnz, pastix_int_t );
        memcpy( graphdst->rows, graphsrc->rows, nnz * sizeof(pastix_int_t) );
    }

    if ( graphsrc->loc2glob != NULL )
    {
        MALLOC_INTERN( graphdst->loc2glob, graphsrc->n, pastix_int_t );
        memcpy( graphdst->loc2glob, graphsrc->loc2glob, graphsrc->n * sizeof(pastix_int_t) );
    }

    if ( graphsrc->glob2loc != NULL )
    {
        MALLOC_INTERN( graphdst->glob2loc, graphsrc->gN, pastix_int_t );
        memcpy( graphdst->glob2loc, graphsrc->glob2loc, graphsrc->gN * sizeof(pastix_int_t) );
    }

    return PASTIX_SUCCESS;
}

/**
 * @}
 */
