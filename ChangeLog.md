# pastix-6.0.2

- Integrate the clusting strategies developped for low-rank (See https://hal.inria.fr/hal-01961675)
- Restructure the ordering/symbolic factorization code to make sure with exit the ordering step with permutation, partition, and elimination tree.
- Relook the splitting/proportional mapping strategy
- Add new compression kernels: PQRCP, RQRCP, and TQRCP
- Fix inplace compilation (Issue #36)
- Fix issue when StarPU threads where fighting for ressources with PaStiX threads (Add pause/resume calls)
- Handle multi-dof static and/or variable in the analysis steps

# pastix-6.0.1

- Support for HWLOC 2.0.0
- Move the SPM library as a submodule
- Parallel (multithreaded) version of the reordering step
- Parallel (multithreaded) version of the refinement steps
- StarPU version of the solve step
- Fix Python/Fortran interface
- Fix Schur functions
- Fix multi-RHS solve
- Fix for METIS
- Update morse_cmake FindPACKAGE
- Update PaRSEC for release 6.0.1
- Improve PKG-CONFIG
- Improve documentation
- Better handle of disconnected graphs
- Add optimal reordering for grids
- Add more detailed statistics during analysis step
- Add detailed statistics about memory gain for the low-rank solver
- Add a function to compress the solver matrix outside the factorization step
- Add an example to dump the symbol matrix including the ranks of the block
- Add an refinement driver and testings
- Add a subtask_refine which does not perform the vector ordering
- Add a more complex testing based on example proposed by @andrea3.14
- Add an iparm IPARM_APPLYPERM_WS to enable/disable the use of an extra workspace to make the functions bvec_xlapmr thread-safe (by default, it is enabled, if disabled, the functions have no memory overhead but loose the thread-safe property)
- Remove the sparse-kit package to avoid conflict (the driver is replaced by HB)

# pastix-6.0.0

- low-rank compression (See https://hal.inria.fr/hal-01824275)
- static scheduler, PaRSEC and StarPU runtime support
- GPUs (Kepler) and KNL support through runtime systems
