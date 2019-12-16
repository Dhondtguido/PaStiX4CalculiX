/*
 * Options.h
 *
 *  Created on: Nov 21, 2014
 *      Author: yongchao
 */

#ifndef OPTIONS_H_
#define OPTIONS_H_

#include "Types.h"

struct Options {
	Options() {

		/*input*/
		_routine = 1;
		_formula = 1;
		_numIters = 1;
		_singlePrecision = false;

		/*matrix data*/
		_numRows = 0;
		_numCols = 0;
		_rowOffsets = NULL;
		_numValues = 0;
		_colIndexValues = NULL;
		_numericalValues = NULL;
		_alpha = 1.0;
		_beta = 1.0;

		/*vector data*/
		_vectorX = NULL;
		_vectorY = NULL;

		/*the number of GPUs*/
		_numGPUs = 1;

		/*GPU index used*/
		_gpuIndex = 0;

		/*for debug*/
		_mean = 0;
		_variance = 0;
	}
	~Options() {/*
		if (_rowOffsets) {
			cudaFreeHost(_rowOffsets);
		}
		if (_colIndexValues) {
			cudaFreeHost(_colIndexValues);
		}
		if (_numericalValues) {
			cudaFreeHost(_numericalValues);
		}

		if (_vectorX) {
			cudaFreeHost(_vectorX);
		}
		if (_vectorY) {
			cudaFreeHost(_vectorY);
		}*/
	}


	/*load vector*/
	/*bool loadVector(const string& fileName, void* vector,
			const int64_t maxNumValues);*/



	/*get row distribution*/
	void getRowSizeVariance();

	/*retrieve GPU list*/
	bool getGPUs();

	/*input files*//*
	string _mmFileName;
	string _vecXFileName;
	string _vecYFileName;
	string _outFileName;*/
	bool _singlePrecision;
	int64_t _routine;
	int64_t _formula;
	int64_t _numIters;
	double _alpha;
	double _beta;

	/*for debugging*/
	double _mean;
	double _variance;

	/*matrix data*/
	int64_t _numRows;
	int64_t _numCols;
	int64_t *_rowOffsets;
	int64_t _numValues;
	int64_t *_colIndexValues;
	void *_numericalValues;

	/*vector data*/
	void *_vectorX;
	void *_vectorY;

	/*number of GPUs to be used*/
	int64_t _numGPUs;

	/*GPU index used*/
	int64_t _gpuIndex;

	/*GPU device list*/
	vector<pair<int64_t, struct cudaDeviceProp> > _gpus;
};

#endif /* OPTIONS_H_ */
