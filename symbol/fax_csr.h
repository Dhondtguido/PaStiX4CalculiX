/**
 *
 * @file fax_csr.h
 *
 * PaStiX fax amalgamation routines
 *
 * @copyright 2004-2018 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @version 6.0.1
 * @author Pascal Henon
 * @author Mathieu Faverge
 * @date 2018-07-16
 *
 * @addtogroup symbol_fax_dev
 * @{
 *
 **/
#ifndef _fax_csr_h_
#define _fax_csr_h_

/**
 * @brief Fax blocked csr structure
 */
typedef struct fax_csr_s {
    pastix_int_t   n;
    pastix_int_t   total_nnz;
    pastix_int_t * nnz;
    pastix_int_t **rows;
} fax_csr_t;

void         faxCSRInit( pastix_int_t n, fax_csr_t *csr );
void         faxCSRClean( fax_csr_t *csr );

pastix_int_t faxCSRGetNNZ( const fax_csr_t *csr );

int  faxCSRGenPA( const pastix_graph_t *graphA, const pastix_int_t *perm, fax_csr_t *graphPA );
void faxCSRCompact( fax_csr_t *csr );

void faxCSRCblkCompress( const fax_csr_t      *graphA,
                         const pastix_order_t *order,
                         fax_csr_t            *graphL,
                         pastix_int_t         *work );

pastix_int_t faxCSRFactDirect( const fax_csr_t      *graphA,
                               const pastix_order_t *order,
                               fax_csr_t            *graphL );
pastix_int_t faxCSRFactILUk( const fax_csr_t      *graphA,
                             const pastix_order_t *order,
                             pastix_int_t          level,
                             fax_csr_t            *graphL );

void faxCSRAmalgamate( int             ilu,
                       double          rat_cblk,
                       double          rat_blas,
                       fax_csr_t      *graphL,
                       pastix_order_t *order,
                       MPI_Comm        pastix_comm );

/**
 * @}
 */
#endif /* _fax_csr_h_ */
