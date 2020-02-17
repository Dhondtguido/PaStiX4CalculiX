/**
 *
 * @file bvec.c
 *
 * @copyright 2004-2018 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @version 6.0.1
 * @author Mathieu Faverge
 * @author Pierre Ramet
 * @author Vincent Bridonneau
 * @date 2018-07-16
 *
 **/
#include "common.h"
#include "bvec.h"


#include <parsec.h>
#include <parsec/data.h>
#include <parsec/data_distribution.h>
#if defined(PASTIX_WITH_CUDA)
#include <parsec/devices/cuda/dev_cuda.h>
#endif
#include "parsec/utils/zone_malloc.h"

extern gpu_device_t* gpu_device;
extern char* gpu_base;

/**
 *******************************************************************************
 *
 * @ingroup bcsc
 *
 * @brief Allocate a vector
 *
 *******************************************************************************
 *
 * @param[in] size
 *          The size of the vector
 *
 *******************************************************************************
 *
 * @return The allocated vector
 *
 *******************************************************************************/
void *bvec_malloc( size_t size )
{
    void *x = NULL;
    MALLOC_INTERN(x, size, char);
    return x;
}

/**
 *******************************************************************************
 *
 * @ingroup bcsc
 *
 * @brief Allocate a vector
 *
 *******************************************************************************
 *
 * @param[in] size
 *          The size of the vector
 *
 *******************************************************************************
 *
 * @return The allocated vector
 *
 *******************************************************************************/
void *bvec_malloc_cuda( size_t size )
{
	
	
#ifdef PASTIX_WITH_CUDA
    void *x = gpu_base;
    //x = zone_malloc(gpu_device->memory, sizeof(char) * size);
    //cudaMalloc(&x, sizeof(char) * size);
    gpu_base += size;
    return x;
#endif
}

/**
 *******************************************************************************
 *
 * @ingroup bcsc
 *
 * @brief Free a vector
 *
 *******************************************************************************
 *
 * @param[inout] x
 *          The vector to be free
 *
 *******************************************************************************/
void bvec_free( void *x )
{
    memFree_null(x);
}

/**
 *******************************************************************************
 *
 * @ingroup bcsc
 *
 * @brief Free a vector
 *
 *******************************************************************************
 *
 * @param[inout] x
 *          The vector to be free
 *
 *******************************************************************************/
void bvec_free_cuda( void *x )
{
	(void) x;/*
#ifdef PASTIX_WITH_CUDA
    cudaFree( x );
    //zone_free( gpu_device->memory, x );
#endif*/
}

