/*
 * SpMV.h
 *
 *  Created on: Nov 21, 2014
 *      Author: yongchao
 */

#ifndef SPMV_H_
#define SPMV_H_
#include "Options.h"
#include "sys/time.h"

class SpMV {
public:
	SpMV(Options* opt);
	virtual ~SpMV() = 0;

	/*compute the number of threads per block*/
	inline void getKernelGridInfo(const int64_t dev,
			int64_t & numThreadsPerBlock, int64_t &numThreadBlocks) {

		/*set to the maximum number of threads per block*/
		numThreadsPerBlock = _opt->_gpus[dev].second.maxThreadsPerBlock;

		/*set to the number of multiprocessors*/
		numThreadBlocks = _opt->_gpus[dev].second.multiProcessorCount
				* (_opt->_gpus[dev].second.maxThreadsPerMultiProcessor
						/ numThreadsPerBlock);

		//cerr << numThreadsPerBlock << " " << numThreadBlocks << endl;
	}

	inline double getSysTime() {
		double dtime;
		struct timeval tv;

		/*get the time of the day*/
		gettimeofday(&tv, NULL);

		/*get the milli-seconds*/
		dtime = ((double) tv.tv_sec) * 1000.0;
		dtime += ((double) tv.tv_usec) / 1000.0;

		return dtime;
	}
	void spmvKernel();
	virtual void loadData() = 0;
	virtual void storeData() = 0;

	/*y = AX*/
	virtual void invokeKernel(const int64_t i) = 0;
	/*y = alpha * Ax + beta * y*/
	virtual void invokeKernelBLAS(const int64_t i) = 0;

	/*member variable*/
	Options* _opt;

	/*number of GPUs*/
	int64_t _numGPUs;

	/*average number of elements per row*/
	int64_t _meanElementsPerRow;

	/*stream*/
	vector<cudaStream_t> _streams;

	/*row counter*/
	vector<unsigned long long*> _cudaRowCounters;
};

/*use global memory*/
/*vector-based row dynamic distribution*/
class SpMVFloatVector: public SpMV {
public:
	SpMVFloatVector(Options* opt);
	virtual ~SpMVFloatVector();

	void loadData();
	void storeData();

	/*y = Ax*/
	virtual void invokeKernel(const int64_t i);
	/*y = alpha * Ax + beta * y*/
	virtual void invokeKernelBLAS(const int64_t i);

	vector<int64_t*> _rowOffsets;
	vector<int64_t*> _colIndexValues;
	vector<float*> _numericalValues;
	vector<float*> _vectorY;
	vector<float*> _vectorX;

	float _alpha;
	float _beta;
};

/*warp-based row dynamic distribution*/
class SpMVFloatWarp: public SpMVFloatVector {
public:
	SpMVFloatWarp(Options* opt) :
			SpMVFloatVector(opt) {
	}

	/*y = Ax*/
	void invokeKernel(const int64_t i);
	/*y = alpha * Ax + beta * y*/
	void invokeKernelBLAS(const int64_t i);
};

class SpMVDoubleVector: public SpMV {
public:
	SpMVDoubleVector(Options* opt);
	virtual ~SpMVDoubleVector();

	void loadData();
	void storeData();

	/*y = Ax*/
	virtual void invokeKernel(const int64_t i);

	/*y = alpha * Ax + beta * y*/
	virtual void invokeKernelBLAS(const int64_t i);

	vector<int64_t*> _rowOffsets;
	vector<int64_t*> _colIndexValues;
	vector<double*> _numericalValues;
	vector<double*> _vectorY;
	vector<double*> _vectorX;

	double _alpha;
	double _beta;
};

/*warp-based row dynamic distribution*/
class SpMVDoubleWarp: public SpMVDoubleVector {
public:
	SpMVDoubleWarp(Options* opt) :
			SpMVDoubleVector(opt) {
	}
	/*y = Ax*/
	void invokeKernel(const int64_t i);

	/*y = alpha * Ax + beta * y*/
	void invokeKernelBLAS(const int64_t i);
};
#endif /* SPMV_H_ */
