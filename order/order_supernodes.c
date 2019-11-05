/**
 *
 * @file order_supernodes.c
 *
 *   PaStiX order routines dedicated to split supernodes thanks to graph connectivity
 *
 * @copyright 2004-2018 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @version 6.0.0
 * @author Gregoire Pichon
 * @date 2018-11-08
 *
 */
#include "common.h"
#include <string.h>
#include "graph.h"
#include "pastix/order.h"
#include "blend/elimintree.h"
#include "blend/extendVector.h"

/**
 *******************************************************************************
 *
 * @ingroup pastix_order
 *
 * @brief Order the supernodes with one of the clustering strategies.
 *
 *******************************************************************************
 *
 * @param[in] graph
 *          The graph that represents the sparse matrix.
 *
 * @param[inout] order
 *          The graph that represents the sparse matrix. On exit, this ordering
 *          is updated with the refined partition obtained thanks to one of the
 *          clustering strategies.
 *
 * @param[inout] etree
 *          The elimination tree. On exit it can be modified by
 *          eTreeComputeLevels().
 *
 * @param[in] iparm
 *          The integer array of parameters.
 *
 *******************************************************************************
 *
 * @retval PASTIX_SUCCESS on successful exit,
 * @retval PASTIX_ERR_BADPARAMETER if one parameter is incorrect,
 *
 *******************************************************************************/
pastix_int_t
orderSupernodes( const pastix_graph_t *graph,
                 pastix_order_t       *order,
                 EliminTree           *etree,
                 pastix_int_t         *iparm )
{
    ExtendVectorINT sn_parts;
    pastix_split_t  strategy     = iparm[IPARM_SPLITTING_STRATEGY];
    pastix_int_t    max_depth    = iparm[IPARM_SPLITTING_PROJECTIONS_DEPTH];
    pastix_int_t    max_distance = iparm[IPARM_SPLITTING_PROJECTIONS_DISTANCE];
    pastix_int_t    max_width    = iparm[IPARM_SPLITTING_PROJECTIONS_WIDTH];
    pastix_int_t    lvl_kway     = iparm[IPARM_SPLITTING_LEVELS_KWAY];
    pastix_int_t    lvl_proj     = iparm[IPARM_SPLITTING_LEVELS_PROJECTIONS];
    pastix_int_t   *depth_size = NULL;
    int            *n_levels   = NULL;
    pastix_int_t    i, sn_first, sn_id;
    pastix_int_t    new_cblknbr;
    pastix_int_t   *perm_treetab;
    pastix_int_t   *new_rangtab;
    pastix_int_t   *new_treetab;
    int8_t         *new_selevtx;

    if ( strategy == PastixSplitNot ) {
        return PASTIX_SUCCESS;
    }

    if ( order == NULL ) {
        errorPrint ("orderSupernodes: invalid order pointer");
        return PASTIX_ERR_BADPARAMETER;
    }

    pastixOrderBase( order, 0 );

    /* Make sure the node levels are computed in the etree */
    eTreeComputeLevels( etree, eTreeRoot(etree), 0 );

    /* Get the minimal index of a node at the max depth considered */
    sn_first = eTreeGetLevelMinIdx( etree, eTreeRoot(etree), lvl_proj+lvl_kway, order->cblknbr );

    extendint_Init( &sn_parts, 16 );

    /*
     * Create an array with the depth of each vertex in the elimination tree
     * used to compute the projection
     */
    if ( strategy == PastixSplitKwayProjections )
    {
        /* Allocate the size of each projection */
        MALLOC_INTERN( depth_size, max_depth, pastix_int_t );

        /* Store the level of each node */
        MALLOC_INTERN( n_levels, graph->n, int );
        for( sn_id=0; sn_id<order->cblknbr; sn_id++ ) {
            int level = etree->nodetab[sn_id].ndlevel;
            assert( level > 0 );

            for(i=order->rangtab[sn_id]; i<order->rangtab[sn_id+1]; i++) {
                n_levels[ order->peritab[i] ] = level;
            }
        }
    }

    /* Backup initial rangtab and allocated the permutation array for top elements of the treetab */
    MALLOC_INTERN( perm_treetab, order->cblknbr - sn_first, pastix_int_t );
    MALLOC_INTERN( new_rangtab,  order->vertnbr+1,          pastix_int_t );
    MALLOC_INTERN( new_treetab,  order->vertnbr,            pastix_int_t );
    MALLOC_INTERN( new_selevtx,  order->vertnbr,            int8_t );

    memcpy( new_rangtab, order->rangtab, sizeof(pastix_int_t) * (sn_first+1) );
    memcpy( new_treetab, order->treetab, sizeof(pastix_int_t) *  sn_first    );
    memset( new_selevtx, 0, order->vertnbr * sizeof(int8_t) );

    new_cblknbr = sn_first;

    for (sn_id = sn_first; sn_id < order->cblknbr; sn_id++) {
        pastix_graph_t sn_graph;
        pastix_order_t sn_order;
        pastix_int_t  sn_level, sn_vertnbr;
        pastix_int_t  fnode, lnode, ret; /* , sorted; */
        pastix_int_t  sn_nbpart_proj, sn_nbparts, sn_nbparts_max;

        perm_treetab[sn_id-sn_first] = new_cblknbr;

        /* sorted     = 0; */
        fnode      = order->rangtab[sn_id];
        lnode      = order->rangtab[sn_id+1];
        sn_vertnbr = lnode - fnode;
        sn_level   = etree->nodetab[sn_id].ndlevel;
        sn_nbparts_max = pastix_iceil( sn_vertnbr, iparm[IPARM_MAX_BLOCKSIZE] );

        if ( (sn_level > (lvl_kway+lvl_proj)) ||
             (sn_nbparts_max == 1) )
        {
            new_treetab[ new_cblknbr ] = order->treetab[sn_id];
            new_cblknbr++;
            new_rangtab[ new_cblknbr ] = lnode;
            continue;
        }

        if ( iparm[IPARM_VERBOSE] > 2 ) {
            fprintf( stdout, " - Working on cblk %ld (level= %d, n= %ld):\n",
                     (long)sn_id, (int)sn_level, (long)(lnode-fnode) );
        }

        /* Reinitialize date structures */
        extendint_Clear( &sn_parts );

        pastixOrderAllocId( &sn_order, sn_vertnbr );
        memset( &sn_graph, 0, sizeof(pastix_graph_t) );

        /**
         * Extract the subgraph with unknowns of the supernode sn_id
         */
        ret = graphIsolateRange( graph, order, &sn_graph,
                                 fnode, lnode, max_distance );
        if ( ret != EXIT_SUCCESS ) {
            fprintf(stderr, "Fatal error in graphIsolateSupernode()!\n");
            exit(1);
        }
        assert( sn_vertnbr == sn_graph.n );

        /**
         * Compute sets of preselected unknowns based on projections
         */
        if ( ( sn_vertnbr >  (16 * iparm[IPARM_MAX_BLOCKSIZE]) ) &&
             ( sn_level   <= lvl_proj )                          &&
             ( strategy == PastixSplitKwayProjections )          &&
             ( etree->nodetab[ sn_id ].sonsnbr == 2 ) )
        {
            memset( depth_size, 0, max_depth * sizeof(pastix_int_t) );

            pastix_int_t *permtab = malloc(graph->n * sizeof(pastix_int_t));
            pastix_int_t *peritab = malloc(graph->n * sizeof(pastix_int_t));
            memcpy(permtab, order->permtab, graph->n * sizeof(pastix_int_t));
            memcpy(peritab, order->peritab, graph->n * sizeof(pastix_int_t));

            graphComputeProjection( graph, n_levels, order,
                                    &sn_graph, &sn_order,
                                    fnode, lnode, sn_level,
                                    1, max_depth, max_width,
                                    depth_size );

            /* Print statistics */
            if ( iparm[IPARM_VERBOSE] > 2 )
            {
                fprintf(stdout, "    - Results of the projection:\n" );
                for( i=0; i<max_depth; i++ ) {
                    fprintf( stdout, "      - At level %d: %8ld\n",
                             (int)(i+1), (long)depth_size[i] );
                }
            }

            /* Partition the supernode if sets of preselected unknowns have correct sizes */
            {
                pastix_int_t selected, total, totalsel;
                pastix_int_t totalmax = iparm[IPARM_MAX_BLOCKSIZE] * 16;
                pastix_int_t selecmax = 50 * sqrt( sn_vertnbr );
                selected = 0;
                totalsel = 0;
                total    = sn_vertnbr;

                for( i=0; i<max_depth; i++ ) {
                    totalsel += depth_size[i];
                    if ( totalsel > selecmax ) {
                        break;
                    }

                    total    -= depth_size[i];
                    selected += depth_size[i];

                    if ( (selected > iparm[IPARM_MIN_BLOCKSIZE]) && (total > totalmax) ) {
                        extendint_Add( &sn_parts, selected );
                        selected = 0;
                    }
                }

                if ( iparm[IPARM_VERBOSE] > 2 ) {
                    fprintf( stdout, "    - %ld nodes selected, %ld remains for K-Way\n",
                             (long)(sn_vertnbr - total - selected), (long)(total + selected) );
                }

                /*
                 * The number of remaining unknowns is the non-selected unknowns
                 * + the selected unknows not extracted
                 */
                if ((sn_vertnbr - total - selected) == 0){
                    memcpy(order->permtab, permtab, graph->n * sizeof(pastix_int_t));
                    memcpy(order->peritab, peritab, graph->n * sizeof(pastix_int_t));
                }

                sn_vertnbr = total + selected;

            }
            free(permtab);
            free(peritab);
        }

        sn_nbparts = extendint_Size( &sn_parts );
        sn_nbpart_proj = sn_nbparts;

        /* Ordering based on K-way */
        if ( ( strategy == PastixSplitKway ) ||
             ( strategy == PastixSplitKwayProjections ) )
        {
            pastix_queue_t queue;
            pastix_int_t *comp_vtx, *comp_sze;
            pastix_int_t comp_nbr = 1;
            pastix_int_t nbpart_kway;
            pastix_int_t smallcp_id, smallcp_sz, cp_id, cp_sz;
            nbpart_kway = pastix_iceil( sn_vertnbr, iparm[IPARM_MAX_BLOCKSIZE] );

            /* Quick return */
            if ( nbpart_kway < 2 ) {
                goto cblk_end;
            }

            /* Update the subgraph by removing the selected vertices if any */
            if ( sn_nbparts > 0 ) {
                pastix_graph_t tmpgraph;
                memset( &tmpgraph, 0, sizeof(pastix_graph_t) );

                /*
                 * We isolate with a distance 0 here, as we already reconnected
                 * the graph at a given distance previously. In that case, we
                 * really want to disconnect components that are connected
                 * though selected vertices
                 */
                graphIsolateRange( &sn_graph, &sn_order, &tmpgraph,
                                   0, sn_vertnbr, 0 );
                graphExit( &sn_graph );
                memcpy( &sn_graph, &tmpgraph, sizeof(pastix_graph_t) );

                /*
                 * Reduce the suborder structure.
                 * We use thefact that the main peritab as always been updated
                 * along with the subarrays.
                 */
                pastixOrderExit( &sn_order );
                pastixOrderAllocId( &sn_order, sn_vertnbr );
            }

            /* Isolate the connected components */
            comp_vtx = malloc( sn_vertnbr * sizeof(pastix_int_t) );
            comp_sze = malloc( sn_vertnbr * sizeof(pastix_int_t) );

            comp_nbr = graphIsolateConnectedComponents( &sn_graph, comp_vtx, comp_sze );

            if ( iparm[IPARM_VERBOSE] > 2 ) {
                fprintf(stdout, "    - Connected components: %ld\n", (long)comp_nbr );
            }

            pqueueInit( &queue, comp_nbr );
            for( i=0; i<comp_nbr; i++) {
                pqueuePush1( &queue, i, comp_sze[i] );
            }

            smallcp_id = -1;
            smallcp_sz = 0;
            while( pqueueSize( &queue ) > 0 ) {
                cp_id = pqueuePop( &queue );
                cp_sz = comp_sze[cp_id];

                if ( cp_sz < iparm[IPARM_COMPRESS_MIN_WIDTH] ) {
                    /* Merge with other small components */
                    smallcp_sz += cp_sz;
                    if ( smallcp_id == -1 ) {
                        smallcp_id = cp_id;
                    }
                    else {
                        comp_sze[cp_id] = 0;
                        for( i=0; (i<sn_graph.n) && (cp_sz>0); i++) {
                            if ( comp_vtx[i] == cp_id ) {
                                comp_vtx[i] = smallcp_id;
                                cp_sz--;
                            }
                        }
                    }
                    comp_sze[ smallcp_id ] = smallcp_sz;
                }
                else {
                    /* Update the local number of K-Way */
                    nbpart_kway = pastix_iceil( cp_sz, iparm[IPARM_MAX_BLOCKSIZE] );
                    if ( nbpart_kway < 2 ) {
                        continue;
                    }

                    /* if (!sorted) { */
                    /*     void *sortptr[3]; */
                    /*     pastix_int_t *perm = sn_order.permtab; */
                    /*     pastix_int_t *invp = sn_order.peritab; */

                    /*     sortptr[0] = comp_vtx; */
                    /*     sortptr[1] = order->peritab + fnode; */
                    /*     sortptr[2] = invp; */

                    /*     qsort3IntAsc( sortptr, sn_graph.n ); */

                    /*     /\* Update the perm array *\/ */
                    /*     for(i=0; i<sn_graph.n; i++) { */
                    /*         perm[invp[i]] = i; */
                    /*     } */

                    /*     sorted = 1; */
                    /* } */

                    graphComputeKway( &sn_graph, &sn_order, order->peritab + fnode,
                                      &comp_nbr, comp_sze, comp_vtx,
                                      cp_id, nbpart_kway );
                }
            }
            pqueueExit( &queue );

            /* If multiple partitions, let's sort the unknowns */
            if ( comp_nbr > 1 ) {
                void *sortptr[2];
                sortptr[0] = comp_vtx;
                sortptr[1] = order->peritab + fnode;

                qsort2IntAsc( sortptr, sn_graph.n );
            }

            for(i=comp_nbr-1; i>=0; i--) {
                if (comp_sze[i] > 0) {
                    extendint_Add( &sn_parts, comp_sze[i] );
                }
            }
            sn_vertnbr = 0;

            free( comp_vtx );
            free( comp_sze );

        }

        /* Let's add a first cblk with remaining nodes */
      cblk_end:
        if ( sn_vertnbr > 0 ) {
            extendint_Add( &sn_parts, sn_vertnbr );
        }

        sn_nbparts = extendint_Size( &sn_parts );
        assert( sn_nbparts >= 1 );
        assert( new_rangtab[ new_cblknbr ] == fnode );

        /* First cblk */
        fnode += extendint_Read( &sn_parts, sn_nbparts-1 );
        new_selevtx[ new_cblknbr ] = ( sn_nbparts-1 < sn_nbpart_proj ) ? 1 : 0;
        new_cblknbr++;
        new_rangtab[ new_cblknbr ] = fnode;

        assert( extendint_Read( &sn_parts, sn_nbparts-1 ) > 0 );
        assert( fnode <= lnode );

        /*
         * Update rangtab and treetab
         */
        for(i=sn_nbparts-2; i>=0; i--)
        {
            /* Chain cblk together */
            new_treetab[new_cblknbr-1] = -1 - new_cblknbr;
            new_selevtx[ new_cblknbr ] = ( i < sn_nbpart_proj ) ? 1 : 0;
            fnode += extendint_Read( &sn_parts, i );
            new_cblknbr++;
            new_rangtab[new_cblknbr] = fnode;

            assert( extendint_Read( &sn_parts, i ) > 0 );
            assert( fnode <= lnode );
        }

        new_treetab[ new_cblknbr-1 ] = order->treetab[sn_id];

        /* Update permtab for future extractions */
        for (i=order->rangtab[sn_id]; i<order->rangtab[sn_id+1]; i++){
            order->permtab[order->peritab[i]] = i;
        }

        graphExit( &sn_graph );
        pastixOrderExit( &sn_order );
    }
    assert( new_rangtab[new_cblknbr] == order->vertnbr );

    if ( n_levels != NULL ) {
        free( n_levels );
    }
    if ( depth_size != NULL ) {
        free( depth_size );
    }
    extendint_Exit( &sn_parts );

    /* Update the treetab */
    {
        pastix_int_t *oldtree, *newtree;

        memFree_null( order->treetab );
        MALLOC_INTERN( order->treetab, new_cblknbr, pastix_int_t );

        newtree = order->treetab;
        oldtree = new_treetab;
        for(i=0; i<new_cblknbr; i++, newtree++, oldtree++) {
            if ( *oldtree >= sn_first ) {
                *newtree = perm_treetab[ *oldtree - sn_first ];
            }
            else if ( *oldtree >= 0 ) {
                *newtree = *oldtree;
            }
            else if ( *oldtree == -1 ) {
                *newtree = -1;
            }
            else { /* < -1 */
                /* Use the fact that 0 will never be a father to shift only by one to escape the root */
                *newtree = - *oldtree - 1;
            }
        }
        memFree_null( new_treetab );
        memFree_null( perm_treetab );
    }

    /* Update the rangtab */
    free(order->rangtab);
    new_rangtab = realloc( new_rangtab, (new_cblknbr+1) * sizeof( pastix_int_t ) );
    order->rangtab = new_rangtab;

    order->cblknbr = new_cblknbr;
    order->selevtx = realloc( new_selevtx, new_cblknbr * sizeof(int8_t) );

    if ( pastixOrderCheck( order ) != 0 ) {
        printf("pastixOrderCheck() at the end of OrderSupernodes() failed !!!");
        assert(0);
    }

    return PASTIX_SUCCESS;
}
