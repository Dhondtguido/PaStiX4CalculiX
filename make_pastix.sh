#!/bin/bash
if ! [[ -d build ]]; then
    mkdir build
fi
cd build

INSTALLPATH="~/pastix"
CUDADIR="/usr/lib/cuda"
PARSECDIR="/usr/lib/parsec"
SCOTCHDIR="/usr/lib/scotch"
HWLOCDIR="/usr/lib/hwloc"

cmake \
	-DCUDA_TOOLKIT_ROOT_DIR=${CUDADIR} \
	-DCMAKE_INSTALL_PREFIX=${INSTALLPATH} \
	-DCMAKE_BUILD_TYPE=Release \
	-DPASTIX_WITH_PARSEC=ON \
	-DPARSEC_DIR=${PARSECDIR} \
	-DSCOTCH_DIR=${SCOTCHDIR} \
	-DHWLOC_DIR=${HWLOCDIR} \
	-DPASTIX_WITH_CUDA=ON \
	-DCUDA_DIR=${CUDADIR} \
	-DPASTIX_ORDERING_SCOTCH=ON \
	-DCMAKE_C_COMPILER=gcc \
	-DCMAKE_CXX_COMPILER=g++ \
	-DCMAKE_Fortran_COMPILER=gfortran \
	-DCMAKE_C_FLAGS="-fopenmp" \
    ..

make -j8
make install
