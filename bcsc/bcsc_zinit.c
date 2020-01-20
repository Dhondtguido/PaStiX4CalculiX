/**
 *
 * @file bcsc_zinit.c
 *
 * @copyright 2004-2018 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @version 6.0.1
 * @author Mathieu Faverge
 * @author Pierre Ramet
 * @author Xavier Lacoste
 * @date 2018-07-16
 *
 * @precisions normal z -> c d s
 *
 **/
#include "common.h"
#include "pastix/order.h"
#include "spm.h"
#include "solver.h"
#include "bcsc.h"
#include "common/cpp_sort.h"
#include "bcsc_z.h"

void transpose_z_Matrix(pastix_int_t 		n,
						pastix_int_t* 		colptrIn, 
						pastix_int_t* 		rowptrIn, 
						cuDoubleComplex*	valuesIn, 
						cuDoubleComplex* 	valuesOut){
	
	pastix_int_t* temp = (pastix_int_t*) calloc(n, sizeof(pastix_int_t));
	
	for(int i = 0; i < n; i++){
		for(int j = colptrIn[i] - 1; j < colptrIn[i+1] - 1 ; j++){
			pastix_int_t target = colptrIn[rowptrIn[j]-1]-1 + (temp[rowptrIn[j]-1]++);
			valuesOut[target] = valuesIn[j];
		}
	}
	
	free(temp);
}

void permute_z_Matrix(pastix_int_t		n,
					  pastix_int_t* 	colptrIn,
					  pastix_int_t* 	rowptrIn, 
					  cuDoubleComplex* 	valuesIn, 
					  pastix_int_t* 	perm, 
					  pastix_int_t* 	peri, 
					  pastix_int_t* 	colptrOut, 
					  pastix_int_t* 	rowptrOut, 
					  cuDoubleComplex*  valuesOut){
	
	colptrOut[0] = 1;
	for(pastix_int_t i = 0; i < n; i++){
		colptrOut[i+1] = colptrOut[i] + colptrIn[peri[i]+1] - colptrIn[peri[i]];
	}
	
	#pragma omp parallel for
	for(pastix_int_t i = 0; i < n; i++){
		for(pastix_int_t j = colptrIn[i] - 1; j < colptrIn[i+1] - 1 ; j++){
			pastix_int_t target = colptrOut[perm[i]] - 1 + j - colptrIn[i] + 1;
			
			rowptrOut[target] = perm[rowptrIn[j]-1] + 1;
			valuesOut[target] = valuesIn[j];
		}
	}
}

/**
 *******************************************************************************
 *
 * @ingroup bcsc_internal
 *
 * @brief Initialize the values in the block csc stored in the given spm.
 *
 *******************************************************************************
 *
 * @param[in] spm
 *          The initial sparse matrix in the spm format.
 *
 * @param[in] ord
 *          The ordering that needs to be applied on the spm to generate the
 *          block csc.
 *
 * @param[in] solvmtx
 *          The solver matrix structure that describe the data distribution.
 *
 * @param[in] col2cblk
 *          Array of matching column with cblk indexes.
 *
 * @param[inout] bcsc
 *          On entry, the pointer to an allocated bcsc.
 *          On exit, the bcsc values field is updated.
 *
 *******************************************************************************/
static inline void
bcsc_zinit_A( const spmatrix_t     *spm,
              const pastix_order_t *ord,
              const SolverMatrix   *solvmtx,
              const pastix_int_t   *col2cblk,
                    pastix_bcsc_t  *bcsc )
{
    pastix_complex64_t *values  = (pastix_complex64_t*)(spm->values);
    pastix_complex64_t *Lvalues = (pastix_complex64_t*)(bcsc->Lvalues);
    pastix_int_t itercblk, itercol, baseval;
    pastix_int_t i, ival, idofcol, idofrow;
    int dof = spm->dof;

    baseval = spm->colptr[0];

    /**
     * Initialize the values of the matrix A in the blocked csc format. This
     * applies the permutation to the values array.
     */
     
    for (itercol=0; itercol<spm->gN; itercol++)
    {
        pastix_int_t *coltab;
        pastix_int_t  fcolnum, frow, lrow;
        pastix_int_t  itercol2 = ord->permtab[itercol] * dof;
        itercblk = col2cblk[ itercol2 ];

        /* The block column is not stored locally, we skip it */
        if (itercblk == -1)
            continue;

        coltab  = bcsc->cscftab[itercblk].coltab;
        fcolnum = solvmtx->cblktab[itercblk].fcolnum;

        frow = spm->colptr[itercol]   - baseval;
        lrow = spm->colptr[itercol+1] - baseval;

        for (i=frow; i<lrow; i++)
        {
            pastix_int_t iterrow  = spm->rowptr[i]-baseval;
            pastix_int_t iterrow2 = ord->permtab[iterrow] * dof;
            
            /*ival = i * dof * dof;

            for (idofcol = 0; idofcol < dof; idofcol++)
            {
                pastix_int_t colidx = itercol2 + idofcol - fcolnum;
                pastix_int_t rowidx = iterrow2;
                pastix_int_t pos = coltab[ colidx ];

                for (idofrow = 0; idofrow < dof;
                     idofrow++, ival++, rowidx++, pos++)
                {
                    bcsc->rowtab[ pos ] = rowidx;
                    Lvalues[ pos ] = values[ ival ];
                }

                coltab[ colidx ] += dof;
                assert( coltab[ colidx ] <= coltab[ colidx+1 ] );
            }*/

			pastix_int_t colidx = itercol2 + 0 - fcolnum;
			pastix_int_t pos = coltab[ colidx ];

			bcsc->rowtab[ pos ] = iterrow2;
			Lvalues[ pos ] = values[ i ];
			
			coltab[ colidx ] ++;
			//assert( coltab[ colidx ] <= coltab[ colidx+1 ] );
        }
    }
}

/**
 *******************************************************************************
 *
 * @ingroup bcsc_internal
 *
 * @brief Initialize the values in the block csc (upper part) for a symmetric
 * matrix since only one side has been initialized by bcsc_zinit_A()
 *
 *******************************************************************************
 *
 * @param[in] spm
 *          The initial sparse matrix in the spm format.
 *
 * @param[in] ord
 *          The ordering that needs to be applied on the spm to generate the
 *          block csc.
 *
 * @param[in] solvmtx
 *          The solver matrix structure that describe the data distribution.
 *
 * @param[in] col2cblk
 *          Array of matching column with cblk indexes.
 *
 * @param[inout] bcsc
 *          On entry, the pointer to an allocated bcsc.
 *          On exit, the bcsc values field is updated.
 *
 *******************************************************************************/
static inline void
bcsc_zinit_Lt( const spmatrix_t     *spm,
               const pastix_order_t *ord,
               const SolverMatrix   *solvmtx,
               const pastix_int_t   *col2cblk,
                     pastix_bcsc_t  *bcsc )
{
    pastix_complex64_t *values  = (pastix_complex64_t*)(spm->values);
    pastix_complex64_t *Lvalues = (pastix_complex64_t*)(bcsc->Lvalues);
    pastix_int_t itercblk, itercol, baseval;
    pastix_int_t i, ival, idofcol, idofrow;
    int dof = spm->dof;

    baseval = spm->colptr[0];

    /**
     * Initialize the values of the matrix A^t in the blocked csc format. This
     * applies the permutation to the values array.
     */
    for (itercol=0; itercol<spm->gN; itercol++)
    {
        pastix_int_t frow, lrow;
        pastix_int_t itercol2 = ord->permtab[itercol] * dof;

        frow = spm->colptr[itercol]   - baseval;
        lrow = spm->colptr[itercol+1] - baseval;

        for (i=frow; i<lrow; i++)
        {
            pastix_int_t *coltab;
            pastix_int_t fcolnum;
            pastix_int_t iterrow  = spm->rowptr[i]-baseval;
            pastix_int_t iterrow2 = ord->permtab[iterrow] * dof;

            itercblk = col2cblk[ iterrow2 ];

            /* The block column is not stored locally, we skip it */
            if ((itercblk == -1) || (iterrow == itercol))
                continue;

            coltab  = bcsc->cscftab[itercblk].coltab;
            fcolnum = solvmtx->cblktab[itercblk].fcolnum;

            ival = i * dof * dof;

            for (idofcol = 0; idofcol < dof; idofcol++)
            {
                pastix_int_t colidx = itercol2 + idofcol;
                pastix_int_t rowidx = iterrow2 - fcolnum;
                pastix_int_t pos;

                for (idofrow = 0; idofrow < dof;
                     idofrow++, ival++, rowidx++, pos++)
                {
                    pos = coltab[ rowidx ];

                    bcsc->rowtab[ pos ] = colidx;
                    Lvalues[ pos ] = values[ ival ];

                    coltab[ rowidx ]++;
                }
            }
        }
    }
}

/**
 *******************************************************************************
 *
 * @ingroup bcsc_internal
 *
 * @brief Initialize the values in the block csc (upper part) for an hermitian
 * matrix since only one side has been initialized by bcsc_zinit_A()
 *
 *******************************************************************************
 *
 * @param[in] spm
 *          The initial sparse matrix in the spm format.
 *
 * @param[in] ord
 *          The ordering that needs to be applied on the spm to generate the
 *          block csc.
 *
 * @param[in] solvmtx
 *          The solver matrix structure that describe the data distribution.
 *
 * @param[in] col2cblk
 *          Array of matching column with cblk indexes.
 *
 * @param[inout] bcsc
 *          On entry, the pointer to an allocated bcsc.
 *          On exit, the bcsc values field is updated.
 *
 *******************************************************************************/
#if defined(PRECISION_z) || defined(PRECISION_c)
static inline void
bcsc_zinit_Lh( const spmatrix_t     *spm,
               const pastix_order_t *ord,
               const SolverMatrix   *solvmtx,
               const pastix_int_t   *col2cblk,
                     pastix_bcsc_t  *bcsc )
{
    pastix_complex64_t *values  = (pastix_complex64_t*)(spm->values);
    pastix_complex64_t *Lvalues = (pastix_complex64_t*)(bcsc->Lvalues);
    pastix_int_t itercblk, itercol, baseval;
    pastix_int_t i, ival, idofcol, idofrow;
    int dof = spm->dof;

    baseval = spm->colptr[0];

    /**
     * Initialize the values of the matrix A^t in the blocked csc format. This
     * applies the permutation to the values array.
     */
    for (itercol=0; itercol<spm->gN; itercol++)
    {
        pastix_int_t frow, lrow;
        pastix_int_t itercol2 = ord->permtab[itercol] * dof;

        frow = spm->colptr[itercol]   - baseval;
        lrow = spm->colptr[itercol+1] - baseval;

        for (i=frow; i<lrow; i++)
        {
            pastix_int_t *coltab;
            pastix_int_t fcolnum;
            pastix_int_t iterrow  = spm->rowptr[i]-baseval;
            pastix_int_t iterrow2 = ord->permtab[iterrow] * dof;

            itercblk = col2cblk[ iterrow2 ];

            /* The block column is not stored locally, we skip it */
            if ((itercblk == -1) || (iterrow == itercol))
                continue;

            coltab  = bcsc->cscftab[itercblk].coltab;
            fcolnum = solvmtx->cblktab[itercblk].fcolnum;

            ival = i * dof * dof;

            for (idofcol = 0; idofcol < dof; idofcol++)
            {
                pastix_int_t colidx = itercol2 + idofcol;
                pastix_int_t rowidx = iterrow2 - fcolnum;
                pastix_int_t pos;

                for (idofrow = 0; idofrow < dof;
                     idofrow++, ival++, rowidx++, pos++)
                {
                    pos = coltab[ rowidx ];

                    bcsc->rowtab[ pos ] = colidx;
                    Lvalues[ pos ] = conj( values[ ival ] );

                    coltab[ rowidx ]++;
                }
            }
        }
    }
}
#endif /* defined(PRECISION_z) || defined(PRECISION_c) */

/**
 *******************************************************************************
 *
 * @brief Initialize a value array with the transpose of A that will be used to
 * initialize the coeftab arrays.
 *
 *******************************************************************************
 *
 * @param[in] spm
 *          The initial sparse matrix in the spm format.
 *
 * @param[in] ord
 *          The ordering that needs to be applied on the spm to generate the
 *          block csc.
 *
 * @param[in] solvmtx
 *          The solver matrix structure that describe the data distribution.
 *
 * @param[in] col2cblk
 *          Array of matching column with cblk indexes.
 *
 * @param[out] trowtab
 *          The row tab associated to the transposition of A.
 *
 * @param[inout] bcsc
 *          On entry, the pointer to an allocated bcsc.
 *          On exit, the bcsc Uvalues field is updated.
 *
 *******************************************************************************/
void
bcsc_zinit_At( const spmatrix_t     *spm,
               const pastix_order_t *ord,
               const SolverMatrix   *solvmtx,
               const pastix_int_t   *col2cblk,
                     pastix_int_t   *trowtab,
                     pastix_bcsc_t  *bcsc )
{
    pastix_complex64_t *values  = (pastix_complex64_t*)(spm->values);
    pastix_complex64_t *Uvalues = (pastix_complex64_t*)(bcsc->Uvalues);
    pastix_int_t itercblk, itercol, baseval;
    pastix_int_t i, ival, idofcol, idofrow;
    int dof = spm->dof;

    baseval = spm->colptr[0];

    /**
     * Initialize the values of the matrix A^t in the blocked csc format. This
     * applies the permutation to the values array.
     */
    for (itercol=0; itercol<spm->gN; itercol++)
    {
        pastix_int_t frow, lrow;
        pastix_int_t itercol2 = ord->permtab[itercol] * dof;

        frow = spm->colptr[itercol]   - baseval;
        lrow = spm->colptr[itercol+1] - baseval;

        for (i=frow; i<lrow; i++)
        {
            pastix_int_t *coltab;
            pastix_int_t fcolnum;
            pastix_int_t iterrow  = spm->rowptr[i]-baseval;
            pastix_int_t iterrow2 = ord->permtab[iterrow] * dof;

            itercblk = col2cblk[ iterrow2 ];

            /* The block column is not stored locally, we skip it */
            if (itercblk == -1)
                continue;

            coltab  = bcsc->cscftab[itercblk].coltab;
            fcolnum = solvmtx->cblktab[itercblk].fcolnum;

            ival = i * dof * dof;

            for (idofcol = 0; idofcol < dof; idofcol++)
            {
                pastix_int_t colidx = itercol2 + idofcol;
                pastix_int_t rowidx = iterrow2 - fcolnum;
                pastix_int_t pos;

                for (idofrow = 0; idofrow < dof;
                     idofrow++, ival++, rowidx++)
                {
                    pos = coltab[ rowidx ];

                    trowtab[ pos ] = colidx;
                    Uvalues[ pos ] = values[ ival ];

                    coltab[ rowidx ]++;
                }
            }
        }
    }
}

/**
 *******************************************************************************
 *
 * @ingroup bcsc_internal
 *
 * @brief Sort the block csc subarray associated to each column block
 *
 *******************************************************************************
 *
 * @param[in] bcsc
 *          On entry, the pointer to an allocated bcsc.
 *
 * @param[in] rowtab
 *          The initial sparse matrix in the spm format.
 *
 * @param[in] valtab
 *          The ordering that needs to be applied on the spm to generate the
 *          block csc.
 *
 *******************************************************************************/

void bcsc_zsort( pastix_bcsc_t *bcsc,
            pastix_int_t        **rowtab,
            pastix_complex64_t  **valtab,
            pastix_int_t  		**sorttab)
{
    bcsc_cblk_t *blockcol;
    pastix_int_t itercblk, itercol, size;
    
    pastix_complex64_t* permedValues;
    pastix_int_t* permedRows;
    
	MALLOC_INTERN(permedValues, bcsc->numElements, pastix_complex64_t);
	MALLOC_INTERN(permedRows, bcsc->numElements, pastix_int_t);

    blockcol = bcsc->cscftab;
    
    if(*sorttab == NULL){
		MALLOC_INTERN(*sorttab, bcsc->numElements, pastix_int_t);
		
		#pragma omp parallel for
		for(int i = 0; i < bcsc->numElements; i++){
			(*sorttab)[i] = i;
		}
		
		for (itercblk=0; itercblk<bcsc->cscfnbr; itercblk++, blockcol++)
		{
			#pragma omp parallel for
			for (itercol=0; itercol<blockcol->colnbr; itercol++)
			{
				/*size = blockcol->coltab[itercol+1] - blockcol->coltab[itercol];
				for(int i = blockcol->coltab[itercol]; i < blockcol->coltab[itercol+1]; i++){
					printf("sortTab[%d] = %ld\n", i, (bcsc->sortTab)[i]);
				}
				printf("\n _______________ \n");
				
				for(int i = blockcol->coltab[itercol]; i < blockcol->coltab[itercol+1]; i++){
					printf("rows[%d] = %ld\n", i, (*rowtab)[i]);
				}
				printf("\n _______________ \n");
*/
				cppSort( (*sorttab) + blockcol->coltab[itercol], (*sorttab) + blockcol->coltab[itercol+1], *rowtab );
				/*
				for(int i = blockcol->coltab[itercol]; i < blockcol->coltab[itercol+1]; i++){
					printf("rows[%d] = %ld\n", i, permedRows[i]);
				}
				printf("\n _______________ \n");
				for(int i = blockcol->coltab[itercol]; i < blockcol->coltab[itercol+1]; i++){
					printf("sortTab[%d] = %ld\n", i, (bcsc->sortTab)[i]);
				}
				printf("\n\n\n");
				
				if(itercol == 1)
					exit(0);*/
			}
		}
	}
    #pragma omp parallel for
    for(int i = 0; i < bcsc->numElements; i++){
		permedValues[i] = (*valtab)[(*sorttab)[i]];
		permedRows[i] = (*rowtab)[(*sorttab)[i]];
	}
    
    memFree_null(*rowtab);
    memFree_null(*valtab);
    
    *rowtab = permedRows;
    *valtab = permedValues;
    
   /* blockcol = bcsc->cscftab;
    for (itercblk=0; itercblk<bcsc->cscfnbr; itercblk++, blockcol++)
    {
        for (itercol=0; itercol<blockcol->colnbr; itercol++)
        {            
            for(int i = blockcol->coltab[itercol]; i < blockcol->coltab[itercol+1]; i++){
				printf("row[%d] = %ld\n", i, (*rowtab)[i]);
			}
			printf("\n\n\n");
			if(itercol == 1)
				exit(0);
        }
    }*/
}

/**
 *******************************************************************************
 *
 * @brief Initialize a centralize pastix_complex64_t block csc.
 *
 *******************************************************************************
 *
 * @param[in] spm
 *          The initial sparse matrix in the spm format.
 *
 * @param[in] ord
 *          The ordering that needs to be applied on the spm to generate the
 *          block csc.
 *
 * @param[in] solvmtx
 *          The solver matrix structure that describe the data distribution.
 *
 * @param[in] col2cblk
 *          Array of matching column with cblk indexes.
 *
 * @param[in] initAt
 *          A flag to enable/disable the initialization of A'
 *
 * @param[inout] bcsc
 *          On entry, the pointer to an allocated bcsc.
 *          On exit, the bcsc stores the input spm with the permutation applied
 *          and grouped accordingly to the distribution described in solvmtx.
 *
 *******************************************************************************/
void
bcsc_zinit_centralized( const spmatrix_t     *spm,
                        const pastix_order_t *ord,
                        const SolverMatrix   *solvmtx,
                        const pastix_int_t   *col2cblk,
                              int             initAt,
                              pastix_bcsc_t  *bcsc )
{		
	pastix_int_t valuesize = bcsc->numElements;
    /**
     * Initialize the blocked structure of the matrix A
     */
    bcsc_zinit_A( spm, ord, solvmtx, col2cblk, bcsc );
	
    if ( spm->mtxtype == SpmSymmetric ) {
        bcsc_zinit_Lt( spm, ord, solvmtx, col2cblk, bcsc );
    }
#if defined(PRECISION_z) || defined(PRECISION_c)
    else if ( spm->mtxtype == SpmHermitian ) {
        bcsc_zinit_Lh( spm, ord, solvmtx, col2cblk, bcsc );
    }
#endif /* defined(PRECISION_z) || defined(PRECISION_c) */

    /* Restore the correct coltab arrays */
    bcsc_restore_coltab( bcsc );

    /* Sort the csc */
    bcsc_zsort( bcsc, &(bcsc->rowtab), &(bcsc->Lvalues), &(bcsc->sorttab) );

    if ( spm->mtxtype == SpmGeneral ) {
	/* A^t is not required if only refinement is performed */
        if (initAt) {
            pastix_int_t *trowtab, i;
            if(!bcsc->Uvalues)
				MALLOC_INTERN( bcsc->Uvalues, valuesize * pastix_size_of( bcsc->flttype ), char );
            MALLOC_INTERN( trowtab, valuesize, pastix_int_t);

            for (i=0; i<valuesize; i++) {
                trowtab[i] = -1;
            }

            bcsc_zinit_At( spm, ord, solvmtx, col2cblk, trowtab, bcsc );

            /* Restore the correct coltab arrays */
            bcsc_restore_coltab( bcsc );

	    /* Sort the transposed csc */
	    bcsc_zsort( bcsc, &trowtab, &(bcsc->Uvalues), &(bcsc->sorttab) );
	    memFree( trowtab );
        }
    }
    else {
        /* In case of SpmHermitian, conj is applied when used to save memory space */
        bcsc->Uvalues = bcsc->Lvalues;
    }
}
