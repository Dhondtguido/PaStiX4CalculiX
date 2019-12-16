#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC void createLightSpMV(int64_t m, int64_t nnz);
EXTERNC void performLightLsMV(
    double alpha,
    double* dval,
    int64_t* drowptr,
    int64_t* dcolind,
    double* dx,
    double beta,
    double* dy);
EXTERNC void destroyLightSpMV();

#undef EXTERNC
