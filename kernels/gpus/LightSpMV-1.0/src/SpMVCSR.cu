/*
 * SpMVCSR.cu
 *
 *  Created on: Nov 25, 2014
 *      Author: yongchao
 */
#include "SpMVCSR.h"

/*device variables*/
__constant__ int64_t _cudaNumRows;
__constant__ int64_t _cudaNumCols;

