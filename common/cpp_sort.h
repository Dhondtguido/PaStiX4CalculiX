#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif
#include <pastix.h>

EXTERNC void cppSort(pastix_int_t * indicesBegin, pastix_int_t* indicesEnd, pastix_int_t* arr);

#undef EXTERNC
