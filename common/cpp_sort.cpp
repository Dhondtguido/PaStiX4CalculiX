#include "cpp_sort.h"

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif
#include <cstddef>
#include <iostream>
#include <algorithm>

class sort_indices
{
   private:
     pastix_int_t* mparr;
   public:
     sort_indices(pastix_int_t* parr) : mparr(parr) {}
     bool operator()(pastix_int_t i, pastix_int_t j) const { /*std::cout << mparr[i] << " < " << mparr[j] << std::endl; */return mparr[i]<mparr[j]; }
};

EXTERNC void cppSort(pastix_int_t* indicesBegin, pastix_int_t* indicesEnd, pastix_int_t* arr){
	std::sort(indicesBegin, indicesEnd, sort_indices(arr));
}

#undef EXTERNC
