/**
 *
 * @file frobeniusupdate.h
 *
 * Forumla to update frobenius norm computation in a safe manner.
 *
 * @copyright 2004-2020 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @version 6.0.3
 * @author Mathieu Faverge
 * @date 2019-12-12
 *
 */
#ifndef _frobeniusupdate_h_
#define _frobeniusupdate_h_

/**
 *******************************************************************************
 *
 * @ingroup pastix_internal
 *
 * @brief Update the couple (scale, sumsq) with one element when computing the
 * Frobenius norm.
 *
 * The frobenius norm is equal to scale * sqrt( sumsq ), this method allows to
 * avoid overflow in the sum square computation.
 *
 *******************************************************************************
 *
 * @param[inout] scale
 *           On entry, the former scale
 *           On exit, the update scale to take into account the value
 *
 * @param[inout] sumsq
 *           On entry, the former sumsq
 *           On exit, the update sumsq to take into account the value
 *
 * @param[in] value
 *          The value to integrate into the couple (scale, sumsq)
 *
 *******************************************************************************/
#if defined(PRECISION_d) || defined(PRECISION_z)
static inline void
frobenius_update( int nb, double *scale, double *sumsq, const double *value )
{
    double absval = fabs(*value);
    double ratio;
    if ( absval != 0. ){
        if ( (*scale) < absval ) {
            ratio = (*scale) / absval;
            *sumsq = (double)nb + (*sumsq) * ratio * ratio;
            *scale = absval;
        } else {
            ratio = absval / (*scale);
            *sumsq = (*sumsq) + (double)nb * ratio * ratio;
        }
    }
}

/**
 *******************************************************************************
 *
 * @ingroup pastix_internal
 *
 * @brief Merge together two sum square stored as a couple (scale, sumsq).
 *
 * The frobenius norm is equal to scale * sqrt( sumsq ), this method allows to
 * avoid overflow in the sum square computation.
 *
 *******************************************************************************
 *
 * @param[in] scl_in
 *           The scale factor of the first couple to merge
 *
 * @param[in] ssq_in
 *           The sumsquare factor of the first couple to merge
 *
 * @param[inout] scl_out
 *           On entry, the scale factor of the second couple to merge
 *           On exit, the updated scale factor.
 *
 * @param[inout] ssq_out
 *           The sumsquare factor of the second couple to merge
 *           On exit, the updated sumsquare factor.
 *
 *******************************************************************************/
static inline void
frobenius_merge( double scl_in, double ssq_in,
                 double *scl_out, double *ssq_out )
{
    double ratio;
    if ( (*scl_out) < scl_in ) {
        ratio  = (*scl_out) / scl_in;
        *ssq_out = (*ssq_out) * ratio * ratio + ssq_in;
        *scl_out = scl_in;
    }
    else {
        ratio  = scl_in / (*scl_out);
        *ssq_out = (*ssq_out) + ssq_in * ratio * ratio;
    }
}

#elif defined(PRECISION_s) || defined(PRECISION_c)
static inline void
frobenius_update( int nb, float *scale, float *sumsq, const float *value )
{
    float absval = fabs(*value);
    float ratio;
    if ( absval != 0. ){
        if ( (*scale) < absval ) {
            ratio = (*scale) / absval;
            *sumsq = (float)nb + (*sumsq) * ratio * ratio;
            *scale = absval;
        } else {
            ratio = absval / (*scale);
            *sumsq = (*sumsq) + (float)nb * ratio * ratio;
        }
    }
}

static inline void
frobenius_merge( float scl_in, float ssq_in,
                 float *scl_out, float *ssq_out )
{
    float ratio;
    if ( (*scl_out) < scl_in ) {
        ratio  = (*scl_out) / scl_in;
        *ssq_out = (*ssq_out) * ratio * ratio + ssq_in;
        *scl_out = scl_in;
    }
    else {
        ratio  = scl_in / (*scl_out);
        *ssq_out = (*ssq_out) + ssq_in * ratio * ratio;
    }
}
#endif

#endif /* _frobeniusupdate_h_ */

