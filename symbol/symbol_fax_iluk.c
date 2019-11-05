/**
 *
 * @file symbol_fax_iluk.c
 *
 * PaStiX symbolic factorization routines for ILU(k)
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
#include "symbol.h"
#include "fax_csr.h"

/**
 *******************************************************************************
 *
 * @ingroup symbol_fax_dev
 *
 * @brief Create the symbol matrix from the graph of the non zero pattern of the
 * factorized matrix and the supernode partition.
 *
 *******************************************************************************
 *
 * @param[inout] P
 *          The non zero pattern of the factorized matrix. WARNING: on exit, the
 *          graph is destroyed.
 *
 * @param[in] cblknbr
 *          The number of supernode. Must be equal to P->n.
 *
 * @param[in] rangtab
 *          Integer array of size cblknbr+1.
 *          Contains the supernode partition of the graph.
 *
 * @param[out] symbmtx
 *          On entry, an initialized structure of symbol matrix (see pastixSymbolInit()).
 *          On exit, contains the symbol matrix associated to the graph P and
 *          the supernode partition given.
 *
 *******************************************************************************
 *
 * @retval 0  on success.
 * @retval !0 on failure.
 *
 *******************************************************************************/
int
pastixSymbolFaxILUk( symbol_matrix_t      *symbptr,
                     pastix_int_t          levelk,
                     const pastix_graph_t *graphA,
                     const pastix_order_t *ordeptr )
{
    pastix_int_t  i, j, k, l;
    pastix_int_t  cblknum;
    pastix_int_t  ind;
    pastix_int_t *tmp       = NULL;
    pastix_int_t *node2cblk = NULL;
    pastix_int_t *ja        = NULL;
    pastix_int_t  n, bloknbr, tmpsize;
    pastix_int_t  cblknbr;
    pastix_int_t *rangtab;
    fax_csr_t     graphPA;
    fax_csr_t     graphL;

    assert( ordeptr->baseval == 0 );
    cblknbr = ordeptr->cblknbr;
    rangtab = ordeptr->rangtab;

    n       = rangtab[cblknbr];
    tmpsize = pastix_iceil( n, 2 );
    MALLOC_INTERN( tmp, tmpsize, pastix_int_t );
    MALLOC_INTERN( node2cblk, n, pastix_int_t );

    /*
     * Let's regenerate the compresse graph of L:
     *     1) Apply the permutation
     *     2) Apply the symbolic ILUk
     *     3) Compress the obtained L, with the partition previsouly computed at order stage
     */
    /* Create the graph of P A */
    faxCSRGenPA( graphA, ordeptr->permtab, &graphPA );

    /* Create the graph of L */
    faxCSRFactILUk( &graphPA, ordeptr, levelk, &graphL );
    faxCSRClean( &graphPA );

    assert( cblknbr == graphL.n );

    /**** First we transform the P matrix to find the block ****/
    for ( k = 0; k < cblknbr; k++ ) {
        for ( i = rangtab[k]; i < rangtab[k + 1]; i++ ) {
            node2cblk[i] = k;
        }
    }

    /* Let's update P to store all the couples (frownum,lrownum) in rows arrays */
    bloknbr = 0;
    for ( k = 0; k < cblknbr; k++ ) {
        assert( graphL.nnz[k] >= ( rangtab[k + 1] - rangtab[k] ) );

#if defined( PASTIX_DEBUG_SYMBOL )
        for ( l = 0; l < rangtab[k + 1] - rangtab[k]; l++ ) {
            assert( graphL.rows[k][l] == rangtab[k] + l );
            assert( node2cblk[graphL.rows[k][l]] == k );
        }
#endif

        ja  = graphL.rows[k];
        j   = 0;
        ind = 0;
        while ( j < graphL.nnz[k] ) {
            cblknum = node2cblk[ja[j]];
            l       = j + 1;
            while ( ( l < graphL.nnz[k] ) && ( ja[l] == ja[l - 1] + 1 ) &&
                    ( node2cblk[ja[l]] == cblknum ) ) {
                l++;
            }

            assert( ind < tmpsize );
            tmp[ind++] = ja[j];
            tmp[ind++] = ja[l - 1];
            assert( ( ja[l - 1] - ja[j] + 1 ) == ( l - j ) );

            j = l;
        }

        graphL.nnz[k] = ind;
        bloknbr += ind / 2;

        memFree( graphL.rows[k] );
        MALLOC_INTERN( graphL.rows[k], ind, pastix_int_t );
        memcpy( graphL.rows[k], tmp, ind * sizeof( pastix_int_t ) );
    }

    memFree( tmp );

    assert( bloknbr == faxCSRGetNNZ( &graphL ) / 2 );

#if defined( PASTIX_DEBUG_SYMBOL )
    for ( k = 0; k < cblknbr; k++ ) {
        assert( graphL.nnz[k] > 0 );
        assert( graphL.rows[k][0] == rangtab[k] );
        assert( ( graphL.rows[k][1] - graphL.rows[k][0] + 1 ) == ( rangtab[k + 1] - rangtab[k] ) );
    }
#endif

    /**********************************/
    /*** Compute the symbol matrix ****/
    /**********************************/
    symbptr->baseval = 0;
    symbptr->cblknbr = cblknbr;
    symbptr->bloknbr = bloknbr;
    symbptr->nodenbr = n;
    symbptr->browtab = NULL;

    MALLOC_INTERN( symbptr->cblktab, cblknbr + 1, symbol_cblk_t );
    MALLOC_INTERN( symbptr->bloktab, symbptr->bloknbr, symbol_blok_t );

    ind = 0;
    for ( k = 0; k < cblknbr; k++ ) {
        symbptr->cblktab[k].fcolnum = rangtab[k];
        symbptr->cblktab[k].lcolnum = rangtab[k + 1] - 1;
        symbptr->cblktab[k].bloknum = ind;
        symbptr->cblktab[k].brownum = -1;

        for ( i = 0; i < graphL.nnz[k]; i += 2 ) {
            j                             = graphL.rows[k][i];
            symbptr->bloktab[ind].frownum = j;
            symbptr->bloktab[ind].lrownum = graphL.rows[k][i + 1];
            symbptr->bloktab[ind].lcblknm = k;
            symbptr->bloktab[ind].fcblknm = node2cblk[j];
            ind++;

            assert( node2cblk[j] == node2cblk[graphL.rows[k][i + 1]] );
        }

#if defined( PASTIX_DEBUG_SYMBOL )
        assert( symbptr->bloktab[symbptr->cblktab[k].bloknum].frownum ==
                symbptr->cblktab[k].fcolnum );
        assert( symbptr->bloktab[symbptr->cblktab[k].bloknum].lrownum ==
                symbptr->cblktab[k].lcolnum );
        assert( symbptr->bloktab[symbptr->cblktab[k].bloknum].fcblknm == k );
#endif
    }

    /*  virtual cblk to avoid side effect in the loops on cblk bloks */
    symbptr->cblktab[cblknbr].fcolnum = symbptr->cblktab[cblknbr - 1].lcolnum + 1;
    symbptr->cblktab[cblknbr].lcolnum = symbptr->cblktab[cblknbr - 1].lcolnum + 1;
    symbptr->cblktab[cblknbr].bloknum = ind;
    symbptr->cblktab[cblknbr].brownum = -1;

    assert( ind == symbptr->bloknbr );
    memFree( node2cblk );

    return 0;
}

/**
 *******************************************************************************
 *
 * @ingroup symbol_fax_dev
 *
 * @brief Patch the symbol matrix to add blocks in order to get a
 * real elimination tree.
 *
 * This function is called when ILU(k) factorization is
 * performed and the faxCSRBuildSymbol() function might have returned a symbol
 * matrix that doesn't provide a real elimination tree.
 *
 *******************************************************************************
 *
 * @param[inout] symbmtx
 *          On entry, a generated symbol matrix with faxCSRBuildSymbol() for example.
 *          On exit, the patched symbol matrix with extra blocks to have a real
 *          elimination tree.
 *
 *******************************************************************************/
void
faxCSRPatchSymbol( symbol_matrix_t *symbmtx )
{
    pastix_int_t   i, j, k;
    pastix_int_t  *father     = NULL; /** For the cblk of the symbol matrix **/
    symbol_blok_t *newbloktab = NULL;
    symbol_cblk_t *cblktab    = NULL;
    symbol_blok_t *bloktab    = NULL;
    fax_csr_t      Q;

    cblktab = symbmtx->cblktab;
    bloktab = symbmtx->bloktab;

    MALLOC_INTERN( father, symbmtx->cblknbr, pastix_int_t );
    MALLOC_INTERN( newbloktab, symbmtx->cblknbr + symbmtx->bloknbr, symbol_blok_t );

    faxCSRInit( symbmtx->cblknbr, &Q );

    /*
     * Count how many extra-diagonal bloks are facing each diagonal blok
     * Double-loop to avoid diagonal blocks.
     */
    for ( i = 0; i < symbmtx->cblknbr; i++ ) {
        for ( j = cblktab[i].bloknum + 1; j < cblktab[i + 1].bloknum; j++ ) {
            Q.nnz[bloktab[j].fcblknm]++;
        }
     }

    /* Allocate nFacingBlok integer for each diagonal blok */
    for ( i = 0; i < symbmtx->cblknbr; i++ ) {
        if ( Q.nnz[i] > 0 ) {
            MALLOC_INTERN( Q.rows[i], Q.nnz[i], pastix_int_t );
        }
        else {
            Q.rows[i] = NULL;
        }
    }

    memset( Q.nnz, 0, symbmtx->cblknbr * sizeof( pastix_int_t ) );

    /*
     * Q.rows[k] will contain, for each extra-diagonal facing blok
     * of the column blok k, its column blok.
     */
    for ( i = 0; i < symbmtx->cblknbr; i++ ) {
        for ( j = cblktab[i].bloknum + 1; j < cblktab[i + 1].bloknum; j++ ) {
            k                     = bloktab[j].fcblknm;
            Q.rows[k][Q.nnz[k]++] = i;
        }
    }

    for ( i = 0; i < Q.n; i++ ) {
        father[i] = -1;
    }

    for ( i = 0; i < Q.n; i++ ) {
        /*
         * For each blok facing diagonal blok i, belonging to column blok k.
         */
        for ( j = 0; j < Q.nnz[i]; j++ ) {
            k = Q.rows[i][j];
            assert( k < i );

            while ( ( father[k] != -1 ) && ( father[k] != i ) ) {
                k = father[k];
            }
            father[k] = i;
        }
    }

    for ( i = 0; i < Q.n; i++ ) {
        if ( father[i] == -1 ) {
            father[i] = i + 1;
        }
    }

    faxCSRClean( &Q );

    k = 0;
    for ( i = 0; i < symbmtx->cblknbr - 1; i++ ) {
        pastix_int_t odb, fbloknum;

        fbloknum = cblktab[i].bloknum;
        memcpy( newbloktab + k, bloktab + fbloknum, sizeof( symbol_blok_t ) );
        cblktab[i].bloknum = k;
        k++;
        odb = cblktab[i + 1].bloknum - fbloknum;
        if ( odb <= 1 || bloktab[fbloknum + 1].fcblknm != father[i] ) {
            /** Add a blok toward the father **/
            newbloktab[k].frownum = cblktab[father[i]].fcolnum;
            newbloktab[k].lrownum = cblktab[father[i]].fcolnum; /** OIMBE try lcolnum **/
            newbloktab[k].lcblknm = i;
            newbloktab[k].fcblknm = father[i];
#if defined( PASTIX_DEBUG_SYMBOL )
            if ( father[i] != i ) {
                assert( cblktab[father[i]].fcolnum > cblktab[i].lcolnum );
            }
#endif
            k++;
        }

        if ( odb > 1 ) {
            memcpy( newbloktab + k, bloktab + fbloknum + 1, sizeof( symbol_blok_t ) * ( odb - 1 ) );
            k += odb - 1;
        }
    }

    /** Copy the last one **/
    memcpy( newbloktab + k,
            bloktab + symbmtx->cblktab[symbmtx->cblknbr - 1].bloknum,
            sizeof( symbol_blok_t ) );
    cblktab[symbmtx->cblknbr - 1].bloknum = k;
    k++;
    /** Virtual cblk **/
    symbmtx->cblktab[symbmtx->cblknbr].bloknum = k;

#if defined( PASTIX_DEBUG_SYMBOL )
    assert( k >= symbmtx->bloknbr );
    assert( k < symbmtx->cblknbr + symbmtx->bloknbr );
#endif
    symbmtx->bloknbr = k;
    memFree( symbmtx->bloktab );
    MALLOC_INTERN( symbmtx->bloktab, k, symbol_blok_t );
    memcpy( symbmtx->bloktab, newbloktab, sizeof( symbol_blok_t ) * symbmtx->bloknbr );
    /*  virtual cblk to avoid side effect in the loops on cblk bloks */
    cblktab[symbmtx->cblknbr].bloknum = k;

    memFree( father );
    memFree( newbloktab );
}
