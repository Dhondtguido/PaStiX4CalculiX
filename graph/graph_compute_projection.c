/**
 *
 * @file graph_compute_projection.c
 *
 * PaStiX graph routine to compute projection of lower levels supernodes
 *
 * @copyright 2004-2018 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @version 6.0.1
 * @author Gregoire Pichon
 * @author Pierre Ramet
 * @author Mathieu Faverge
 * @date 2018-07-16
 *
 **/
#include "common.h"
#include "pastix/order.h"
#include "blend/extendVector.h"
#include "graph.h"

void
graphComputeProjection( const pastix_graph_t *graph,
                        const int            *vertlvl,
                              pastix_order_t *order,
                        const pastix_graph_t *subgraph,
                              pastix_order_t *suborder,
                              pastix_int_t    fnode,
                              pastix_int_t    lnode,
                              pastix_int_t    sn_level,
                              pastix_int_t    distance,
                              pastix_int_t    maxdepth,
                              pastix_int_t    maxwidth,
                              pastix_int_t   *depthsze )
{
    ExtendVectorINT vec;
    pastix_int_t baseval = graph->colptr[0];
    pastix_int_t n, i, ip, j, jp, jj, d, k, sze;
    pastix_int_t lvl, depth;
    const pastix_int_t *colptr;
    const pastix_int_t *rows;
    pastix_int_t *peritab, *subvertlvl, *subvert;
    pastix_int_t *perm, *invp;

    n = lnode - fnode;
    MALLOC_INTERN( subvertlvl, n, pastix_int_t );
    extendint_Init( &vec, (sqrt(n)+1) * pastix_imax( distance, maxwidth ) * 2 );

    /* (i, j ) in initial ordering  */
    /* (ip,jp) in permuted ordering */
    colptr  = graph->colptr;
    rows    = graph->rows;
    peritab = order->peritab;
    subvert = subvertlvl;

    for (ip=fnode; ip<lnode; ip++, subvert++) {
        extendint_Clear( &vec );

        /* i^th vertex in the initial numbering */
        extendint_Add( &vec, peritab[ip] );
        *subvert = -maxdepth-1;
        sze =  1;
        d   = -1;
        k   =  0;

        while( d < distance ) {
            for(; k<sze; k++) {
                i = extendint_Read( &vec, k );

                for (jj = colptr[i  ]-baseval;
                     jj < colptr[i+1]-baseval; jj++) {

                    j   = rows[jj]-baseval;
                    lvl = vertlvl[j];

                    /*
                     * If lvl is equal to sn_level, the node belong to the same supernode,
                     * and if lvl is lower than sn_level, then the node belongs to a
                     * supernode higher in the elimination tree.
                     * In both cases, we avoid to connect to a supernode through them.
                     */
                    if ( lvl <= sn_level ) {
                        continue;
                    }

                    /*
                     * Store the negative depth to sort in ascending order with
                     * nodes connected to deeper levels first
                     */
                    depth = sn_level - lvl;
                    *subvert = pastix_imax( depth, *subvert );

                    extendint_Add( &vec, j );
                }
            }
            d++;
            sze = extendint_Size( &vec );
        }
    }

    perm = suborder->permtab;
    invp = suborder->peritab;

    /*
     * Enlarge the projections
     */
    if ( maxwidth > 0 ) {
        pastix_int_t *subvertlv2;
        void *sortptr[3];

        sortptr[0] = subvertlvl;
        sortptr[1] = invp;
        sortptr[2] = order->peritab + fnode;

        qsort3IntAsc( sortptr, n );

        /* Generate the new perm array for the subgraph */
        for(i=0; i<n; i++) {
            j = invp[i];
            assert( (j >= 0) && (j < n) );
            perm[j] = i;
        }

        MALLOC_INTERN( subvertlv2, n, pastix_int_t );
        memcpy( subvertlv2, subvertlvl, n * sizeof(pastix_int_t) );

        colptr = subgraph->colptr;
        rows   = subgraph->rows;
        maxdepth = -maxdepth;

        /*
         * We do the loop in reverse order to first enlarge the nodes connected
         * to the highest supernodes.
         */
        for (ip=n-1; ip>=0; ip--) {
            if ( subvertlv2[ip] < maxdepth ) {
                break;
            }

            extendint_Clear( &vec );
            /* i^th vertex in the initial numbering */
            extendint_Add( &vec, invp[ip] );
            sze =  1;
            d   =  0;
            k   =  0;

            while( d < maxwidth ) {
                for(; k<sze; k++) {
                    i = extendint_Read( &vec, k );
                    assert( subvertlvl[ip] == subvertlvl[ perm[i] ] );

                    for (jj = colptr[i  ];
                         jj < colptr[i+1]; jj++) {

                        j  = rows[jj];
                        jp = perm[j];

                        /* If j has already been seen because connected to an higher sn_level */
                        if ( subvertlv2[jp] > subvertlv2[ip] ) {
                            continue;
                        }
                        subvertlvl[jp] = subvertlvl[ip];
                        extendint_Add( &vec, j );
                    }
                }
                d++;
                sze = extendint_Size( &vec );
            }
        }
        maxdepth = -maxdepth;

        memFree_null( subvertlv2 );
    }

    /* Sort again the peritab array associated to the subgraph */
    {
        void *sortptr[3];
        sortptr[0] = subvertlvl;
        sortptr[1] = invp;
        sortptr[2] = order->peritab + fnode;

        qsort3IntAsc( sortptr, n );

        /* Update the perm array */
        for(i=0; i<n; i++) {
            perm[invp[i]] = i;
        }
    }

    /* Compute the sizes of each depth */
    memset( depthsze, 0, maxdepth * sizeof(pastix_int_t) );
    {
        int d = 0;
        for(i=n-1; i >= 0; i--) {
            while ( subvertlvl[i] < (-d-1) ) {
                d++;
            }
            if ( d >= maxdepth ) {
                break;
            }
            depthsze[d] ++;
        }
    }

    extendint_Exit( &vec );
    free(subvertlvl);
}
