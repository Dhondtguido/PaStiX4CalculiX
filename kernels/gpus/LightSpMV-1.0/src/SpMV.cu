/*
 * SpMV.cu
 *
 *  Created on: Nov 21, 2014
 *      Author: yongchao
 */
#include "SpMV.h"
#include "SpMVCSR.h"

extern __constant__ int64_t _cudaNumRows;

SpMV::SpMV(Options* opt) {
	_opt = opt;

	/*the number of GPUs*/
	_numGPUs = _opt->_numGPUs;

	/*compute the mean number of elements per row*/
	_meanElementsPerRow = (int64_t) rint(
			(double) _opt->_numValues / _opt->_numRows);

	/*create row counter*/
	_cudaRowCounters.resize(_numGPUs, NULL);

	/*create streams*/
	_streams.resize(_numGPUs, 0);

	for (int64_t i = 0; i < _numGPUs; ++i) {
		cudaSetDevice(_opt->_gpus[i].first);
		CudaCheckError();

		cudaStreamCreate(&_streams[i]);
		CudaCheckError();
	}
}
SpMV::~SpMV() {
	/*destroy the streams*/
	for (int64_t i = 0; i < _numGPUs; ++i) {

		/*set device*/
		cudaSetDevice(_opt->_gpus[i].first);
		CudaCheckError();

		cudaStreamDestroy(_streams[i]);
		CudaCheckError();
	}
}

/*invoke kernel*/
void SpMV::spmvKernel() {

	/*initialize the counter*/
	cudaMemset(_cudaRowCounters[0], 0, sizeof(int64_t));

	/*invoke kernel*/
	if (_opt->_formula == 0) {
		invokeKernel(0);
	} else {
		invokeKernelBLAS(0);
	}
}
void SpMV::invokeKernel(const int64_t i) {
	/*do nothing*/
}
void SpMV::invokeKernelBLAS(const int64_t i) {
	/*do nothing*/
}

/*single-precision floating point*/
SpMVFloatVector::SpMVFloatVector(Options* opt) :
		SpMV(opt) {

	_rowOffsets.resize(_numGPUs, NULL);
	_colIndexValues.resize(_numGPUs, NULL);
	_numericalValues.resize(_numGPUs, NULL);
	_vectorY.resize(_numGPUs, NULL);
	_vectorX.resize(_numGPUs, NULL);

	_alpha = _opt->_alpha;
	_beta = _opt->_beta;
}
SpMVFloatVector::~SpMVFloatVector() {
	/*release matrix data*/
	for (int64_t i = 0; i < _numGPUs; ++i) {

		/*select the device*/
		cudaSetDevice(_opt->_gpus[i].first);
		CudaCheckError();

		/*release the resources*/
		if (_rowOffsets[i]) {
			cudaFree(_rowOffsets[i]);
		}
		if (_colIndexValues[i]) {
			cudaFree(_colIndexValues[i]);
		}

		if (_numericalValues[i]) {
			cudaFree(_numericalValues[i]);
		}
		if (i == 0 && _vectorY[i]) {
			cudaFree(_vectorY[i]);
		}
		if (_vectorX[i]) {
			cudaFree(_vectorX[i]);
		}
	}
}
void SpMVFloatVector::loadData() {
	size_t numBytes;

	/*iterate each GPU*/
	for (int64_t i = 0; i < _numGPUs; ++i) {

		/*select the device*/
		cudaSetDevice(_opt->_gpus[i].first);
		CudaCheckError();

		/*allocate counter buffers*/
		cudaMalloc(&_cudaRowCounters[i], sizeof(int64_t));
		CudaCheckError();

		cudaMemcpyToSymbol(_cudaNumRows, &_opt->_numRows, sizeof(int64_t));
		CudaCheckError();

		cudaMemcpyToSymbol(_cudaNumCols, &_opt->_numCols, sizeof(int64_t));
		CudaCheckError();

		/******************************************************
		 * Load matrix data
		 ******************************************************/
		numBytes = (_opt->_numRows + 1) * sizeof(int64_t);
		cudaMalloc(&_rowOffsets[i], numBytes);
		CudaCheckError();

		cudaMemcpy(_rowOffsets[i], _opt->_rowOffsets, numBytes,
				cudaMemcpyHostToDevice);
		CudaCheckError();

		numBytes = _opt->_numValues * sizeof(int64_t);
		cudaMalloc(&_colIndexValues[i], numBytes);
		CudaCheckError();

		cudaMemcpy(_colIndexValues[i], _opt->_colIndexValues, numBytes,
				cudaMemcpyHostToDevice);
		CudaCheckError();

		/*load the numerical values*/
		numBytes = _opt->_numValues * sizeof(float);
		cudaMalloc(&_numericalValues[i], numBytes);
		CudaCheckError();

		cudaMemcpy(_numericalValues[i], _opt->_numericalValues, numBytes,
				cudaMemcpyHostToDevice);
		CudaCheckError();

		/*****************************************************
		 * Load vector X data
		 ******************************************************/
		numBytes = _opt->_numCols * sizeof(float);
		cudaMalloc(&_vectorX[i], numBytes);
		CudaCheckError();

		cudaMemcpy(_vectorX[i], _opt->_vectorX, numBytes,
				cudaMemcpyHostToDevice);
		CudaCheckError();

		/*****************************************************
		 * vector Y data
		 ******************************************************/
		numBytes = _opt->_numRows * sizeof(float);
		cudaMalloc(&_vectorY[i], numBytes);
		CudaCheckError();

		/*copy the data*/
		cudaMemcpy(_vectorY[i], _opt->_vectorY, numBytes,
				cudaMemcpyHostToDevice);
		CudaCheckError();
	}
}
void SpMVFloatVector::storeData() {
	
}
void SpMVFloatVector::invokeKernel(const int64_t i) {
	int64_t numThreadsPerBlock;
	int64_t numThreadBlocks;

	/*get the number of threads per block*/
	getKernelGridInfo(i, numThreadsPerBlock, numThreadBlocks);

	/*invoke the kernel*/

	if (_meanElementsPerRow <= 2) {
		spmv_csr::csr32DynamicVector<float, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	} else if (_meanElementsPerRow <= 4) {
		spmv_csr::csr32DynamicVector<float, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	} else if(_meanElementsPerRow <= 64) {
		spmv_csr::csr32DynamicVector<float, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	} else {
		spmv_csr::csr32DynamicVector<float, 32, MAX_NUM_THREADS_PER_BLOCK / 32><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	}
}

void SpMVFloatVector::invokeKernelBLAS(const int64_t i) {
	int64_t numThreadsPerBlock;
	int64_t numThreadBlocks;

	/*get the number of threads per block*/
	getKernelGridInfo(i, numThreadsPerBlock, numThreadBlocks);


	if (_meanElementsPerRow <= 2) {
		spmv_csr::csr32DynamicVectorBLAS<float, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _alpha, _beta);
	} else if (_meanElementsPerRow <= 4) {
		spmv_csr::csr32DynamicVectorBLAS<float, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _alpha, _beta);
	} else if(_meanElementsPerRow <= 64) {
		spmv_csr::csr32DynamicVectorBLAS<float, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _alpha, _beta);
	} else {
		spmv_csr::csr32DynamicVectorBLAS<float, 32, MAX_NUM_THREADS_PER_BLOCK / 32><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _alpha, _beta);
	}
}

void SpMVFloatWarp::invokeKernel(const int64_t i) {
	int64_t numThreadsPerBlock;
	int64_t numThreadBlocks;

	/*get the number of threads per block*/
	getKernelGridInfo(i, numThreadsPerBlock, numThreadBlocks);


	if (_meanElementsPerRow <= 2) {
		spmv_csr::csr32DynamicWarp<float, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i],_vectorY[i]);
	} else if (_meanElementsPerRow <= 4) {
		spmv_csr::csr32DynamicWarp<float, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	} else if(_meanElementsPerRow <= 64) {
		spmv_csr::csr32DynamicWarp<float, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	} else {
		spmv_csr::csr32DynamicWarp<float, 32, MAX_NUM_THREADS_PER_BLOCK / 32><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	}
}

void SpMVFloatWarp::invokeKernelBLAS(const int64_t i) {
	int64_t numThreadsPerBlock;
	int64_t numThreadBlocks;

	/*get the number of threads per block*/
	getKernelGridInfo(i, numThreadsPerBlock, numThreadBlocks);

	if (_meanElementsPerRow <= 2) {
		spmv_csr::csr32DynamicWarpBLAS<float, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i],_vectorY[i], _alpha, _beta);
	} else if (_meanElementsPerRow <= 4) {
		spmv_csr::csr32DynamicWarpBLAS<float, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _alpha, _beta);
	} else if(_meanElementsPerRow <= 64) {
		spmv_csr::csr32DynamicWarpBLAS<float, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _alpha, _beta);
	} else {
		spmv_csr::csr32DynamicWarpBLAS<float, 32, MAX_NUM_THREADS_PER_BLOCK / 32><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _alpha, _beta);
	}
}

/*double-precision floating point*/
SpMVDoubleVector::SpMVDoubleVector(Options* opt) :
		SpMV(opt) {

	_rowOffsets.resize(_numGPUs, NULL);
	_colIndexValues.resize(_numGPUs, NULL);
	_numericalValues.resize(_numGPUs, NULL);
	_vectorY.resize(_numGPUs, NULL);

	_vectorX.resize(_numGPUs, NULL);

	_alpha = _opt->_alpha;
	_beta = _opt->_beta;

}
SpMVDoubleVector::~SpMVDoubleVector() {
	/*release matrix data*/
	for (int64_t i = 0; i < _numGPUs; ++i) {

		/*select the device*/
		cudaSetDevice(_opt->_gpus[i].first);
		CudaCheckError();

		/*release the resources*/
		if (_rowOffsets[i]) {
			cudaFree(_rowOffsets[i]);
		}
		if (_colIndexValues[i]) {
			cudaFree(_colIndexValues[i]);
		}

		if (_numericalValues[i]) {
			cudaFree(_numericalValues[i]);
		}
		if (i == 0 && _vectorY[i]) {
			cudaFree(_vectorY[i]);
		}
		if (_vectorX[i]) {
			cudaFree(_vectorX[i]);
		}
	}
}
void SpMVDoubleVector::loadData() {
	size_t numBytes;

	/*iterate each GPU*/
	for (int64_t i = 0; i < _numGPUs; ++i) {

		/*select the device*/
		cudaSetDevice(_opt->_gpus[i].first);
		CudaCheckError();

		/*allocate counter buffers*/
		cudaMalloc(&_cudaRowCounters[i], sizeof(int64_t));
		CudaCheckError();

		cudaMemcpyToSymbol(_cudaNumRows, &_opt->_numRows, sizeof(int64_t));
		CudaCheckError();

		cudaMemcpyToSymbol(_cudaNumCols, &_opt->_numCols, sizeof(int64_t));
		CudaCheckError();

		/******************************************************
		 * Load matrix data
		 ******************************************************/
		/*numBytes = (_opt->_numRows + 1) * sizeof(int64_t);
		cudaMalloc(&_rowOffsets[i], numBytes);
		CudaCheckError();

		cudaMemcpy(_rowOffsets[i], _opt->_rowOffsets, numBytes,
				cudaMemcpyHostToDevice);
		CudaCheckError();

		numBytes = _opt->_numValues * sizeof(int64_t);
		cudaMalloc(&_colIndexValues[i], numBytes);
		CudaCheckError();

		cudaMemcpy(_colIndexValues[i], _opt->_colIndexValues, numBytes,
				cudaMemcpyHostToDevice);
		CudaCheckError();*/

		/*load the numerical values*/
		/*numBytes = _opt->_numValues * sizeof(double);
		cudaMalloc(&_numericalValues[i], numBytes);
		CudaCheckError();

		cudaMemcpy(_numericalValues[i], _opt->_numericalValues, numBytes,
				cudaMemcpyHostToDevice);
		CudaCheckError();*/

		/*****************************************************
		 * Load vector X data
		 ******************************************************/
		/*numBytes = _opt->_numCols * sizeof(double);
		cudaMalloc(&_vectorX[i], numBytes);
		CudaCheckError();

		cudaMemcpy(_vectorX[i], _opt->_vectorX, numBytes,
				cudaMemcpyHostToDevice);
		CudaCheckError();*/

		/*****************************************************
		 * vector Y data
		 ******************************************************/
		/*numBytes = _opt->_numRows * sizeof(double);*/
		/*allocate space on the first GPU*/
		/*cudaMalloc(&_vectorY[i], numBytes);
		CudaCheckError();*/

		/*copy the data*/
		/*cudaMemcpy(_vectorY[i], _opt->_vectorY, numBytes,
				cudaMemcpyHostToDevice);
		CudaCheckError();*/
	}
}
void SpMVDoubleVector::storeData() {
	
}
void SpMVDoubleVector::invokeKernel(const int64_t i) {
	int64_t numThreadsPerBlock;
	int64_t numThreadBlocks;

	/*get the number of threads per block*/
	getKernelGridInfo(i, numThreadsPerBlock, numThreadBlocks);

		CudaCheckError();
	/*invoke the kernel*/

	if (_meanElementsPerRow <= 2) {
		spmv_csr::csr64DynamicVector<double, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	} else if (_meanElementsPerRow <= 4) {
		spmv_csr::csr64DynamicVector<double, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	} else if(_meanElementsPerRow <= 64) {
		spmv_csr::csr64DynamicVector<double, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	} else {
		spmv_csr::csr64DynamicVector<double, 32, MAX_NUM_THREADS_PER_BLOCK / 32><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	}
	
		CudaCheckError();
}

void SpMVDoubleVector::invokeKernelBLAS(const int64_t i) {
	int64_t numThreadsPerBlock;
	int64_t numThreadBlocks;

		CudaCheckError();
	/*get the number of threads per block*/
	getKernelGridInfo(i, numThreadsPerBlock, numThreadBlocks);

		CudaCheckError();
	/*invoke the kernel*/

	if (_meanElementsPerRow <= 2) {
		spmv_csr::csr64DynamicVectorBLAS<double, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _vectorY[i], _alpha, _beta);
	} else if (_meanElementsPerRow <= 4) {
		spmv_csr::csr64DynamicVectorBLAS<double, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _vectorY[i], _alpha, _beta);
	} else if(_meanElementsPerRow <= 64) {
		spmv_csr::csr64DynamicVectorBLAS<double, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _vectorY[i], _alpha, _beta);
	} else {
		spmv_csr::csr64DynamicVectorBLAS<double, 32, MAX_NUM_THREADS_PER_BLOCK / 32><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _vectorY[i], _alpha, _beta);
	}
	
		CudaCheckError();
}

void SpMVDoubleWarp::invokeKernel(const int64_t i) {
	int64_t numThreadsPerBlock;
	int64_t numThreadBlocks;

		CudaCheckError();
	/*get the number of threads per block*/
	getKernelGridInfo(i, numThreadsPerBlock, numThreadBlocks);

		CudaCheckError();
	/*invoke the kernel*/

		CudaCheckError();
	if (_meanElementsPerRow <= 2) {
		spmv_csr::csr64DynamicWarp<double, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	} else if (_meanElementsPerRow <= 4) {
		spmv_csr::csr64DynamicWarp<double, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	} else if(_meanElementsPerRow <= 64) {
		spmv_csr::csr64DynamicWarp<double, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	} else {
		spmv_csr::csr64DynamicWarp<double, 32, MAX_NUM_THREADS_PER_BLOCK / 32><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	}

		CudaCheckError();
}

void SpMVDoubleWarp::invokeKernelBLAS(const int64_t i) {
	int64_t numThreadsPerBlock;
	int64_t numThreadBlocks;

	/*get the number of threads per block*/
	getKernelGridInfo(i, numThreadsPerBlock, numThreadBlocks);

	if (_meanElementsPerRow <= 2) {
		spmv_csr::csr64DynamicWarpBLAS<double, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _vectorY[i], _alpha, _beta);
	} else if (_meanElementsPerRow <= 4) {
		spmv_csr::csr64DynamicWarpBLAS<double, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _vectorY[i], _alpha, _beta);
	} else if(_meanElementsPerRow <= 64) {
		spmv_csr::csr64DynamicWarpBLAS<double, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _vectorY[i], _alpha, _beta);
	} else {
		spmv_csr::csr64DynamicWarpBLAS<double, 32, MAX_NUM_THREADS_PER_BLOCK / 32><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _vectorY[i], _alpha, _beta);
	}
}
