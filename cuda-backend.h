// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>
#include "helper_string.h"  // helper for shared functions common to CUDA Samples


class CudaDevice
{

public:
    //function for computing 1st layer of RNN. It chooses a word embedding vector from the vocabulary matrix U and
    //adds it to the recurrent layer h transformed by the matrix V
    void cudaRecurrentLayerCompute(double *io_hiddenActivationDevicePointer,
                                   double *i_hiddenToHiddenMatrixDevicePointer,
                                   double *i_vocabularyMatrixDevicePointer,
                                   int i_hiddenLayerSize,
                                   int i_vocabularySize,
                                   int i_inputWordIndex,
                                   double *i_tmpDevicePointer) const;

    void cudaOutputErrorCompute(double *i_OutputLayerActivationDevicePointer,
                                int i_vecSize,
                                int i_trueWordIndex,
                                double *o_OutputLayerErrorDevicePointer, double tau) const;

    void cudaSoftmaxActivation(double *io_devicePointer,
                               int i_vectorSize) const;

    void cudaVectorByMatrix(double *i_matrixDevicePointer,
                            double *i_vectorDevicePointer,
                            int i_rowsInMat,
                            int i_colsInMat,
                            bool i_transpose,
                            double *o_res,
                            bool i_override = true) const;

    void cudaMatrixOuterProductUpdate(double *i_leftVectorDevicePointer,
                                      double *i_rightVectorDevicePointer,
                                      int i_leftVectorSize,
                                      int i_rightVectorSize,
                                      double i_lr,
                                      double i_beta,
                                      double *i_matrixDeviceMemoryPointer) const;

    double cudaGetVectorCoordinate(double *i_vectorDevicePointer,
                                   int i_coordinate) const;

    void cudaAddVectorToColumn(double *i_vectorDevicePointer,
                               int i_rowNumber, int i_colNumber,
                               int i_column,
                               double i_lr,
                               double i_beta,
                               double *i_matrixDeviceMemoryPointer) const;

    void cudaVectorSum(double *i_leftVectorDevicePointer,
                       double *i_rightVectorDevicePointer,
                       int i_vectorSize,
                       double * o_resDeviceMemoryPointer) const;

    void addMatrixToMatrixAndScale(double *i_matrixDeviceMemoryPointer, int i_nRows,int i_nColumns, double i_beta, double *io_resDeviceMemoryPointer) const;
    void cudaAddColumnToColumn(double *i_matrixDeviceMemoryPointer, int i_nRows, int i_nCols, int i_updateColumn, double *io_resDeviceMemoryPointer, int i_targetColumn, double i_beta) const;

    void logisticErrorActivation(double *i_activationVector, double *io_errorVector, int i_size) const;
    void copy(double *i_source, double *o_destination, int i_destinationSize) const;
    void setZeroColumn(double *iomatrixDeviceMemoryPointer, int i_nRows, int i_nCols, int i_column) const;
    void addMatrixColumnToVector(double *i_matrixDeviceMemoryPointer, int i_nRows, int i_nColumns, int i_column, double *o_vectorDeviceMemoryPointer) const;
    void cudaLogisticActivation(double *io_vector, int i_size) const;
    void setZeroVector(double *io_deviceMemoryPointer, int m_size) const;
    void cudaAddLog(double* i_rvalue, double i_scale, double *o_lvalue) const;
    void cudaAddScalarToScalar(double* i_rvalue, double *o_lvalue) const;

    static const CudaDevice& getDevice()
    {
        static CudaDevice m_device;
        return m_device;
    }

    ~CudaDevice();
private:

    CudaDevice();
    int m_devID;
    int m_blockSize;
    cublasHandle_t m_cublasHandle;
    double* m_deviceBufDouble;
    int* m_deviceBufInt;
    double* m_hostBuf;
};

