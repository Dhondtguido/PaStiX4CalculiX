/*
 * Options.cu
 *
 *  Created on: Nov 24, 2014
 *      Author: yongchao
 */

#include "Options.h"

void Options::getRowSizeVariance() {
	double rowStart;
	int64_t rowEnd;

	/*compute the variance*/
	_variance = 0;
	_mean = rint((double) _numValues / _numRows);
	rowStart = _rowOffsets[0];
	for (int64_t i = 1; i <= _numRows; ++i) {
		rowEnd = _rowOffsets[i];
		_variance += (rowEnd - rowStart - _mean) * (rowEnd - rowStart - _mean);
		rowStart = rowEnd;
	}
	_variance = rint(sqrt(_variance / (_numRows > 1 ? _numRows - 1 : 1)));

	/*information*/
	/*cerr << "Rows: " << _numRows << " Cols: " << _numCols << " Elements: "
			<< _numValues << " Mean: " << _mean << " Standard deviation: "
			<< _variance << endl;*/
}
bool Options::getGPUs() {
	int32_t numGPUs;

	/*get the number of GPUs*/
	if (cudaGetDeviceCount(&numGPUs) != cudaSuccess) {
		//cerr << "No CUDA-enabled GPU is available in the host" << endl;
		return false;
	}


	/*iterate each GPU*/
	cudaDeviceProp prop;
	pair<int64_t, cudaDeviceProp> p;
	for (int64_t i = 0; i < numGPUs; ++i) {

		/*get the property of the device*/
		cudaGetDeviceProperties(&prop, i);

		/*check the major of the GPU*/
		if ((prop.major * 10 + prop.minor) >= 30) {
			/*cerr << "GPU " << _gpus.size() << ": " << prop.name
					<< " (capability " << prop.major << "." << prop.minor << ")"
					<< endl;*/

			/*save the Kepler GPU*/
			p = make_pair(i, prop);
			_gpus.push_back(p);
		}
	}
	/*check the number of qualified GPUs*/
	if (_gpus.size() == 0) {
		//cerr << "No qualified GPU is available" << endl;
		return false;
	}

	/*check the GPU index*/

	/*reset the number of GPUs*/
	if (_gpuIndex >= (int64_t) _gpus.size()) {
		_gpuIndex = _gpus.size() - 1;
	}
	if (_gpuIndex < 0) {
		_gpuIndex = 0;
	}

	/*move the selected gpu to the first*/
	swap(_gpus[0], _gpus[_gpuIndex]);

	return true;
}
