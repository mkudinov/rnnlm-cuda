#include "cuda-backend.h"

#define REDUCE_SIZE 64

__global__ void apply_exp(double *y, int *max_idx, const int Y) {
  __shared__ double AS[REDUCE_SIZE];
  __shared__ int IS[REDUCE_SIZE];

  int bx = blockIdx.x; // from 0 to number of resulting output elements
  int tx = threadIdx.x; // from 0 to BLOCK_SIZE - 1

  // 1. Find the maximum element

  double max_vle = (double)-1.0e30;
  int max_vle_idx = 0;

  int offs = bx * Y + tx, i_offs = tx;
  for (int i = 0; i < Y / REDUCE_SIZE; i++, offs += REDUCE_SIZE, i_offs += REDUCE_SIZE) {
    if (max_vle < y[offs]) {
      max_vle = y[offs];
      max_vle_idx = i_offs;
    }
  }

  if (tx < Y % REDUCE_SIZE) {
    if (max_vle < y[offs]) {
      max_vle = y[offs];
      max_vle_idx = i_offs;
    }
  }

  AS[tx] = max_vle;
  IS[tx] = max_vle_idx;
  // ensure all shared loaded
  __syncthreads();

  // Process found elements
  int n = MIN(Y, REDUCE_SIZE);
  while (n > 1) {
    if (n & 1) {
      if (max_vle < AS[n - 1]) {
        max_vle = AS[n - 1];
        max_vle_idx = IS[n - 1];
      }
    }
    n >>= 1;
    if (tx < n) {
      if (AS[tx] < AS[n + tx]) {
        AS[tx] = AS[n + tx];
        IS[tx] = IS[n + tx];
      }
    }
    // ensure all shared updated
    __syncthreads();
  }
  if (!tx) {
    if (AS[0] < max_vle) {
      AS[0] = max_vle;
      IS[0] = max_vle_idx;
    }
    max_idx[bx] = IS[0];
 //   printf("MAX_IDX: %d\n", IS[0]);
  }
  // ensure all shared updated
  __syncthreads();

  max_vle = AS[0];

  // ensure all shared read 'cause we will update AS later
  __syncthreads();

  // 2. Find the sum(exp(x - max))
  double sum = 0;

  offs = bx * Y + tx;
  for (int i = 0; i < Y / REDUCE_SIZE; i++, offs += REDUCE_SIZE) {
    sum += exp(y[offs] - max_vle);
  }
  // Process the remaining part
  if (tx < Y % REDUCE_SIZE) {
    sum += exp(y[offs] - max_vle);
  }

  AS[tx] = sum;
  // ensure all shared loaded
  __syncthreads();

  // Process found elements
  sum = 0;
  n = MIN(Y, REDUCE_SIZE);
  while (n > 1) {
    if (n & 1) {
      sum += AS[n - 1];
    }
    n >>= 1;
    if (tx < n) {
      AS[tx] += AS[n + tx];
    }
    // ensure all shared summed
    __syncthreads();
  }
  if (!tx) {
    AS[0] += sum;
  }
  // ensure all shared updated
  __syncthreads();

  sum = AS[0];


  // 3. Output exp(x - max) / sum
  offs = bx * Y + tx;
  for (int i = 0; i < Y / REDUCE_SIZE; i++, offs += REDUCE_SIZE) {
    y[offs] = exp(y[offs] - max_vle) / sum;
  }
  // Process the remaining part
  if (tx < Y % REDUCE_SIZE) {
    y[offs] = exp(y[offs] - max_vle) / sum;
 //   printf("%d: %.6f\n", tx, y[offs]);
  }
}


__global__ void logisticActivation(double *io_vec, int num)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < num)
    io_vec[id] = 1/(1+exp(-io_vec[id]));
}

__global__ void getErrorVector(double* i_activationVec, int i_vecLen, int i_targetWord, double *o_errorVec)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id == i_targetWord)
    {
        o_errorVec[id] = 1 - i_activationVec[id];
    }
    else if(id < i_vecLen)
    {
        o_errorVec[id] = -i_activationVec[id];
    }
}

__global__ void softmaxErrorActivationKernel(double *i_activationVector, double *io_errorVector, int i_size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < i_size)
    {
        io_errorVector[id] = io_errorVector[id] * i_activationVector[id] * (1-i_activationVector[id]);
    }
}

__global__ void addLog(double *i_rvalue, double *o_lvalue)
{
    *o_lvalue += log10(*i_rvalue);
}

CudaDevice::CudaDevice()
{
    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    cudaError_t error;
    m_devID = 0;

    // get number of SMs on this GPU
    error = cudaGetDevice(&m_devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaDeviceProp deviceProp;

    error = cudaGetDeviceProperties(&deviceProp, m_devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", m_devID, deviceProp.name, deviceProp.major, deviceProp.minor);

    // use a larger block size for Fermi and above
    m_blockSize = (deviceProp.major < 2) ? 16 : 32;

    checkCudaErrors(cublasCreate(&m_cublasHandle));
    checkCudaErrors(cudaMalloc((void **) &m_deviceBufDouble, sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &m_deviceBufInt, sizeof(int)));
    m_hostBuf = (double *)malloc(sizeof(double));
}

CudaDevice::~CudaDevice()
{
    free(m_hostBuf);
    checkCudaErrors(cudaFree(m_deviceBufInt));
    checkCudaErrors(cudaFree(m_deviceBufDouble));
}

void CudaDevice::cudaVectorByMatrix(double *i_matrixDevicePointer, double *i_vectorDevicePointer, int i_rowsInMat, int i_colsInMat, bool i_transpose, double *o_res) const
{
    const double alpha = 1.0;
    const double beta = 0.0;

    //cublas<t>gemv computes the following expression
    //y = \alpha*A*x + \beta*y
    //because of stupid Fortran-style memory alignment we have to do transpose here to gt normal Matrix x Vector multiplication
    if(!i_transpose)
    //we 1) transpose, 2) i don't know why, but i have to swap rows and columns in both cases
    {
        checkCudaErrors(cublasDgemv(m_cublasHandle, CUBLAS_OP_T, i_colsInMat, i_rowsInMat,  &alpha, i_matrixDevicePointer, i_colsInMat, i_vectorDevicePointer, 1, &beta, o_res, 1));
    }
    else
    {
        checkCudaErrors(cublasDgemv(m_cublasHandle, CUBLAS_OP_N, i_colsInMat, i_rowsInMat,  &alpha, i_matrixDevicePointer, i_colsInMat, i_vectorDevicePointer, 1, &beta, o_res, 1));
    }
}

void CudaDevice::cudaSoftmaxActivation(double *io_devicePointer, int i_vectorSize) const
{
    apply_exp<<<1, REDUCE_SIZE>>>(io_devicePointer, m_deviceBufInt, i_vectorSize);
}

double CudaDevice::cudaGetVectorCoordinate(double *i_vectorDevicePointer, int i_coordinate) const
{
    checkCudaErrors(cudaMemcpy(m_hostBuf, i_vectorDevicePointer + i_coordinate, sizeof(double), cudaMemcpyDeviceToHost));
    return *m_hostBuf;
}

void CudaDevice::cudaOutputErrorCompute(double *i_OutputLayerActivationDevicePointer, int i_vecSize, int i_trueWordIndex, double *o_OutputLayerErrorDevicePointer) const
{
    int gridSize = (int)ceil((float)i_vecSize/m_blockSize);
    getErrorVector<<<gridSize, m_blockSize>>>(i_OutputLayerActivationDevicePointer, i_vecSize, i_trueWordIndex, o_OutputLayerErrorDevicePointer);
}

void CudaDevice::cudaMatrixOuterProductUpdate(double *i_leftVectorDevicePointer,
                                  double *i_rightVectorDevicePointer,
                                  int i_leftVectorSize,
                                  int i_rightVectorSize,
                                  double i_lr,
                                  double i_beta,
                                  double *io_matrixDeviceMemoryPointer) const
{
//    double *one;
//    checkCudaErrors(cudaMalloc((void **) &one, 3 * sizeof(double)));
//    double* initializer =(double *)calloc(3, sizeof(double));
//    for(size_t i = 0; i < 3; i++)
//    {
//        initializer[i] = 2*(i+1);
//    }
//    checkCudaErrors(cudaMemcpy(one, initializer, 3 * sizeof(double), cudaMemcpyHostToDevice));
//    free(initializer);

//    double *two;
//    checkCudaErrors(cudaMalloc((void **) &two, 33 * sizeof(double)));
//    initializer =(double *)calloc(33, sizeof(double));
//    for(size_t i = 0; i < 33; i++)
//    {
//        initializer[i] = 1;
//    }
//    checkCudaErrors(cudaMemcpy(two, initializer, 33 * sizeof(double), cudaMemcpyHostToDevice));
//    free(initializer);

//    double *mat;
//    checkCudaErrors(cudaMalloc((void **) &mat, 3 * 33 * sizeof(double)));
//    initializer =(double *)calloc(3 * 33, sizeof(double));
//    checkCudaErrors(cudaMemcpy(mat, initializer, 3 * 33 * sizeof(double), cudaMemcpyHostToDevice));
//    free(initializer);

//    double *buffer = (double *)malloc(3 * 33 * sizeof(double));
//    checkCudaErrors(cudaMemcpy(buffer, mat, 3 * 33 * sizeof(double), cudaMemcpyDeviceToHost));
//    std::cout << std::endl;
//    for (int i=0; i<3; i++)
//    {
//        for (int j=0; j<33; j++)
//        {
//            std::cout << buffer[j+i*33] << " ";
//        }
//        std::cout << std::endl;
//    }
//    free(buffer);
      double scale = 1-i_beta;
      checkCudaErrors(cublasDgemm(m_cublasHandle,CUBLAS_OP_N,CUBLAS_OP_T,i_rightVectorSize, i_leftVectorSize, 1, &i_lr, i_rightVectorDevicePointer, i_rightVectorSize, i_leftVectorDevicePointer, i_leftVectorSize, &scale, io_matrixDeviceMemoryPointer,i_rightVectorSize));

//    double a = 1;

//    checkCudaErrors(cublasDgemm(m_cublasHandle,CUBLAS_OP_N,CUBLAS_OP_T, 33,3,1,&a,two,33,one,3,&a, mat,33));
//    if(i_beta != 0)
//    {
//        double scale = 1-i_beta;
//      //  checkCudaErrors(cublasDscal(m_cublasHandle, 3, &scale, mat, 33 ));
//    }

//    buffer = (double *)malloc(3 * 33 * sizeof(double));
//        checkCudaErrors(cudaMemcpy(buffer, mat, 3 * 33 * sizeof(double), cudaMemcpyDeviceToHost));
//        std::cout << std::endl;
//        for (int i=0; i<3; i++)
//        {
//            for (int j=0; j<33; j++)
//            {
//                std::cout << buffer[j+i*33] << " ";
//            }
//            std::cout << std::endl;
//        }
//        free(buffer);
}

void CudaDevice::cudaAddVectorToColumn(double *i_leftVectorDevicePointer, int i_rowNumber, int i_colNumber, int i_column, double i_lr, double i_beta, double *io_matrixDeviceMemoryPointer) const
{
     const double scale = 1 - i_beta;
     const double alpha = i_lr;
     checkCudaErrors(cublasDscal(m_cublasHandle, i_rowNumber, &scale, io_matrixDeviceMemoryPointer + i_column, i_colNumber ));
     checkCudaErrors(cublasDaxpy(m_cublasHandle, i_rowNumber, &alpha, i_leftVectorDevicePointer, 1,  io_matrixDeviceMemoryPointer + i_column, i_colNumber));
}

void CudaDevice::softmaxErrorActivation(double *i_activationVector, double *io_errorVector, int i_size) const
{
    int gridSize = (int)ceil((float)i_size/m_blockSize);
    softmaxErrorActivationKernel<<<gridSize,m_blockSize>>>(i_activationVector, io_errorVector, i_size);
}

void CudaDevice::copy(double *i_source, double *o_destination, int i_destinationSize) const
{
    checkCudaErrors(cublasDcopy(m_cublasHandle, i_destinationSize, i_source, 1, o_destination, 1));
}

void CudaDevice::addMatrixToMatrixAndScale(double *i_matrixDeviceMemoryPointer, int i_nRows,int i_nColumns, double i_beta, double *io_resDeviceMemoryPointer) const
{
     const double scale = 1 - i_beta;
     const double alpha = 1;
     checkCudaErrors(cublasDscal(m_cublasHandle, i_nRows * i_nColumns, &scale, io_resDeviceMemoryPointer,1));
     checkCudaErrors(cublasDaxpy(m_cublasHandle, i_nRows * i_nColumns, &alpha, i_matrixDeviceMemoryPointer, 1,  io_resDeviceMemoryPointer, 1));
}

void CudaDevice::cudaAddColumnToColumn(double *i_matrixDeviceMemoryPointer, int i_nRows, int i_nCols, int i_updateColumn, double *io_resDeviceMemoryPointer, int i_targetColumn, double i_beta) const
{
    const double scale = 1 - i_beta;
    const double alpha = 1;
    checkCudaErrors(cublasDscal(m_cublasHandle, i_nRows, &scale, io_resDeviceMemoryPointer + i_targetColumn, i_nCols ));
    checkCudaErrors(cublasDaxpy(m_cublasHandle, i_nRows, &alpha, i_matrixDeviceMemoryPointer + i_updateColumn, i_nCols,  io_resDeviceMemoryPointer + i_targetColumn, i_nCols));
}

void CudaDevice::setZeroColumn(double *io_matrixDeviceMemoryPointer, int i_nRows, int i_nCols, int i_targetColumn) const
{
    const double alpha = 0;
    checkCudaErrors(cublasDscal(m_cublasHandle, i_nRows, &alpha, io_matrixDeviceMemoryPointer + i_targetColumn, i_nCols ));
}

void CudaDevice::cudaVectorSum(double *i_leftVectorDevicePointer, double *i_rightVectorDevicePointer, int i_vectorSize, double *o_resDeviceMemoryPointer) const
{
    const double alpha = 1;
    checkCudaErrors(cublasDcopy(m_cublasHandle, i_vectorSize, i_leftVectorDevicePointer, 1, o_resDeviceMemoryPointer, 1));
    checkCudaErrors(cublasDaxpy(m_cublasHandle, i_vectorSize, &alpha, i_rightVectorDevicePointer, 1,  o_resDeviceMemoryPointer, 1));
}

void CudaDevice::addMatrixColumnToVector(double *i_matrixDeviceMemoryPointer, int i_nRows, int i_nColumns, int i_column, double *o_vectorDeviceMemoryPointer) const
{
    const double alpha = 1;
    //cublas<t>axpy computes the following expression
    //y = y + x*\alpha

    checkCudaErrors(cublasDaxpy(m_cublasHandle, i_nRows, &alpha, i_matrixDeviceMemoryPointer + i_column, i_nColumns,  o_vectorDeviceMemoryPointer, 1));
}

void CudaDevice::cudaLogisticActivation(double *io_vectorDeviceMemoryPointer, int i_size) const
{
    // Number of thread blocks in grid
    int gridSize = (int)ceil((float)i_size/m_blockSize);

    //Activation
    logisticActivation<<<gridSize,m_blockSize>>>(io_vectorDeviceMemoryPointer, i_size);
}

void CudaDevice::setZeroVector(double *io_deviceMemoryPointer, int m_size) const
{
    const double alpha = 0;
    checkCudaErrors(cublasDscal(m_cublasHandle, m_size, &alpha, io_deviceMemoryPointer, 1 ));
}

void CudaDevice::cudaAddLog(double* i_rvalue, double *o_lvalue) const
{
    addLog<<<1,1>>>(i_rvalue, o_lvalue);
}

void CudaDevice::cudaAddScalarToScalar(double* i_rvalue, double *o_lvalue) const
{
    const double alpha = 1;
    checkCudaErrors(cublasDaxpy(m_cublasHandle, 1, &alpha, i_rvalue, 1,  o_lvalue, 1));
}
