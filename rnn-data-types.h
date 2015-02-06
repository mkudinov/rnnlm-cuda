/**
 * @file    rnn-data-types.h
 * @brief	header file for definition of all using data types
 *
 * Copyright 2015 by Samsung Electronics, Inc.,
 * 
 * This software is the confidential and proprietary information
 * of Samsung Electronics, Inc. ("Confidential Information").  You
 * shall not disclose such Confidential Information and shall use
 * it only in accordance with the terms of the license agreement
 * you entered into with Samsung.
 */

#ifndef _RNN_DT_
#define _RNN_DT_

//#define EIGEN_RUNTIME_NO_MALLOC

#include "Eigen/Dense"
#include "Eigen/Sparse"

namespace rnn
{

typedef double ScalarType;
typedef Eigen::SparseMatrix<ScalarType> SparseMat;
typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> DenseMat;
typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> DenseVec;
typedef Eigen::Array<ScalarType, Eigen::Dynamic, Eigen::Dynamic> ArrayXX;
typedef Eigen::Array<ScalarType, Eigen::Dynamic, 1> ArrayX;

}
#endif //_RNN_DT_

