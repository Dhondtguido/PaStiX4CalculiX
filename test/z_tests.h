/**
 *
 * @file z_tests.h
 *
 * Tests functions header.
 *
 * @copyright 2018-2019 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @version 6.0.1
 * @author Gregoire Pichon
 * @author Mathieu Faverge
 * @date 2018-07-16
 *
 * @precisions normal z -> z c d s
 *
 **/
#ifndef _z_tests_h_
#define _z_tests_h_

#include "pastix_lowrank.h"

extern pastix_lr_t z_lowrank;

int z_bcsc_spmv_check( spm_trans_t trans, const spmatrix_t *spm, const pastix_data_t *pastix_data );
int z_bcsc_norm_check( const spmatrix_t   *spm, const pastix_bcsc_t *bcsc );
int z_bvec_gemv_check( int check, int m, int n, pastix_int_t *iparm, pastix_fixdbl_t *dparm );
int z_bvec_check( pastix_data_t *pastix_data, pastix_int_t m );

int z_lowrank_genmat( int mode, double tolerance, pastix_int_t rank,
                      pastix_int_t m, pastix_int_t n,
                      pastix_complex64_t *A, pastix_int_t lda,
                      double             *normA );

int z_lowrank_check_ge2lr( pastix_compress_method_t method, double tolerance,
                           pastix_int_t m, pastix_int_t n,
                           pastix_complex64_t *A, pastix_int_t lda,
                           double normA,
                           fct_ge2lr_t core_zge2lr );

int z_lowrank_check_rradd( pastix_compress_method_t method, double tolerance,
                           pastix_int_t offx, pastix_int_t offy,
                           pastix_int_t mA, pastix_int_t nA,
                           const pastix_lrblock_t *lrA,
                           pastix_int_t mB, pastix_int_t nB,
                           const pastix_lrblock_t *lrB,
                           const pastix_complex64_t *Cfr, pastix_int_t ldc, double normCfr,
                           fct_rradd_t core_rradd );

int z_lowrank_check_lrmm( pastix_compress_method_t method, double tolerance,
                          pastix_int_t offx, pastix_int_t offy,
                          pastix_int_t m, pastix_int_t n, pastix_int_t k,
                          const pastix_lrblock_t *lrA,
                          const pastix_lrblock_t *lrB,
                          pastix_int_t Cm, pastix_int_t Cn,
                          const pastix_lrblock_t *lrC,
                          const pastix_complex64_t *Cfr, pastix_int_t ldc,
                          double normCfr,
                          pastix_lr_t *lowrank );

#endif /* _z_tests_h_ */
