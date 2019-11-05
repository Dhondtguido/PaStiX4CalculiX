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
#include <pastix.h>
#include <spm.h>
#include "graph.h"
#include "symbol.h"
#include "pastix/order.h"
#include "common.h"
#include "solver.h"
#include <scotch.h>

/**
 *******************************************************************************
 *
 * @ingroup pastix_order
 *
 * @brief Dump the last separator into an ivview file.
 *
 *******************************************************************************
 *
 * @param[in] pastix_data
 *          The pastix data structure that holds the graph and the ordering.
 *
 * @param[in] min_cblk
 *          The index of the first vertex belonging to the last separator.
 *
 *******************************************************************************/
void
orderDraw( pastix_data_t *pastix_data,
           pastix_int_t   min_cblk )
{
    FILE           *file;
    pastix_graph_t *graph  = pastix_data->graph;
    pastix_order_t *order  = pastix_data->ordemesh;
    pastix_int_t    size   = order->vertnbr-min_cblk;
    pastix_int_t    color  = 0;
    pastix_int_t    i, j;
    SCOTCH_Graph   sn_sgraph;
    pastix_graph_t sn_pgraph;
    pastix_int_t  *sn_colptr;
    pastix_int_t  *sn_rows;

    graphIsolateRange( graph, order, &sn_pgraph,
                       min_cblk, order->vertnbr,
                       pastix_data->iparm[IPARM_SPLITTING_PROJECTIONS_DISTANCE] );

    sn_colptr = sn_pgraph.colptr;
    sn_rows   = sn_pgraph.rows;


    if(!SCOTCH_graphInit(&sn_sgraph))
    {
        SCOTCH_graphBuild( &sn_sgraph,
                           order->baseval,
                           size,
                           sn_colptr,
                           NULL,
                           NULL,
                           NULL,
                           sn_colptr[ size ] - order->baseval,
                           sn_rows,
                           NULL );
    }
    else {
        fprintf( stderr, "Failed to build graph\n" );
        return;
    }

    file = fopen( "part.grf","w" );
    SCOTCH_graphSave( &sn_sgraph, file );
    fclose(file);

    fprintf(stderr,"Check: %d\n", SCOTCH_graphCheck( &sn_sgraph ));

    free(sn_colptr);
    free(sn_rows);

    /* Build xyz file */
    {
        FILE *fileout;
        int rc;
        (void) rc;
        file = fopen( "before.xyz", "r" );
        if ( file == NULL ) {
            fprintf(stderr, "Please give before.xyz file\n");
            return;
        }

        fileout = fopen( "part.xyz", "w" );

        long dim, n;
        rc = fscanf( file, "%ld %ld", &dim, &n );

        if ( n != order->vertnbr ){
            fprintf(stderr, "Cannot proceed part.xyz and part.map files: invalid number of vertices in before.xyz\n");
            fclose(file);
            fclose(fileout);
            return;
        }

        fprintf(fileout, "%ld %ld\n", (long)dim, (long)size );
        for(i=0; i<order->vertnbr; i++) {
            long v, iv;
            double x, y, z;

            rc = fscanf(file, "%ld %lf %lf %lf", &v, &x, &y, &z );
            assert( rc == 4 );
            /* If permutation in the last supernode, we keep it */
            iv = order->permtab[i];
            if ( iv >= min_cblk ) {
                fprintf(fileout, "%ld %lf %lf %lf\n", (long)(iv-min_cblk), x, y, z);
            }
        }

        fclose(file);
        fclose(fileout);
    }

    /* Set colors */
    {
        FILE *fileout = fopen("part.map" ,"w");
        fprintf(fileout, "%ld\n", (long)size);

        for (i=order->cblknbr-1; i>0; i--){
            pastix_int_t fnode = order->rangtab[i];
            pastix_int_t lnode = order->rangtab[i+1];

            if ( fnode < min_cblk ) {
                assert( lnode <= min_cblk );
                break;
            }

            for (j=fnode; j<lnode; j++) {
                fprintf(fileout, "%ld %ld\n", (long)(j-min_cblk), (long)color);
            }
            color++;
        }
        fclose(fileout);
    }

    /* Free graph structure, we don't need it anymore */
    if (pastix_data->graph != NULL) {
        graphExit( pastix_data->graph );
        memFree_null( pastix_data->graph );
    }
}
