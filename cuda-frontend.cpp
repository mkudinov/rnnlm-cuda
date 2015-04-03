#include <unistd.h>
#include "cuda-frontend.h"

void Vector::setConstant(size_t i_size, double i_constToFill)
{
    m_size  = i_size;
    checkCudaErrors(cudaMalloc((void **) &m_deviceMemoryPointer, i_size * sizeof(double)));
    double* initializer =(double *)calloc(i_size, sizeof(double));
    for(size_t i = 0; i < i_size; i++)
    {
        initializer[i] = i_constToFill;
    }
    checkCudaErrors(cudaMemcpy(m_deviceMemoryPointer, initializer, i_size * sizeof(double), cudaMemcpyHostToDevice));
    free(initializer);
    //m_properties = Properties(m_deviceMemoryPointer, m_size);
    m_buffered.clear();
}

void Vector::setConstant(double i_constToFill)
{
    if(m_size == -1)
    {
        throw ESizeIsUnknown();
    }
    double* initializer =(double *)calloc(m_size, sizeof(double));
    for(int i = 0; i < m_size; i++)
    {
        initializer[i] = i_constToFill;
    }
    checkCudaErrors(cudaMemcpy(m_deviceMemoryPointer, initializer, m_size * sizeof(double), cudaMemcpyHostToDevice));
    free(initializer);
    m_buffered.clear();
}

void Vector::setArray(size_t i_size, double *i_actiivationInitializer)
{
    m_size = i_size;
    checkCudaErrors(cudaMalloc((void **) &m_deviceMemoryPointer, i_size * sizeof(double)));
    double* initializer =(double *)calloc(i_size, sizeof(double));
    checkCudaErrors(cudaMemcpy(m_deviceMemoryPointer, i_actiivationInitializer, i_size * sizeof(double), cudaMemcpyHostToDevice));
    free(initializer);
    m_buffered.clear();
}

void Vector::print() const
{
    double buffer[m_size];
    checkCudaErrors(cudaMemcpy(buffer, m_deviceMemoryPointer, m_size * sizeof(double), cudaMemcpyDeviceToHost));
    std::cout << std::endl;
    for (int i=0; i<m_size; i++)
    {
        std::cout << buffer[i] << std::endl;
    }
}

void Layer::setConstant(size_t i_size, double i_constToFill)
{
    m_size  = i_size;
    ac.setConstant(m_size,i_constToFill);
    er.setConstant(m_size, 0);
}

void Layer::setArray(size_t i_size, double *i_actiivationInitializer)
{
    m_size  = i_size;
    ac.setArray(m_size, i_actiivationInitializer);
    er.setConstant(m_size, 0);
}

void Matrix::setZero(size_t i_nRows, size_t i_nColumns)
{
    m_nColumns = i_nColumns;
    m_nRows = i_nRows;
    checkCudaErrors(cudaMalloc((void **) &m_deviceMemoryPointer, i_nRows * i_nColumns * sizeof(double)));
    double* initializer =(double *)calloc(i_nRows * i_nColumns, sizeof(double));
    checkCudaErrors(cudaMemcpy(m_deviceMemoryPointer, initializer, i_nRows * i_nColumns * sizeof(double), cudaMemcpyHostToDevice));
    free(initializer);
    //m_properties = Properties(m_deviceMemoryPointer,m_nRows,m_nColumns);
    m_buffered.clear();
}

void Matrix::setMatrix(double *i_initializer, size_t i_nRows, size_t i_nColumns)
{
    m_nColumns = i_nColumns;
    m_nRows = i_nRows;
    checkCudaErrors(cudaMalloc((void **) &m_deviceMemoryPointer, i_nRows * i_nColumns * sizeof(double)));
    checkCudaErrors(cudaMemcpy(m_deviceMemoryPointer, i_initializer, i_nRows * i_nColumns * sizeof(double), cudaMemcpyHostToDevice));
    //m_properties = Properties(m_deviceMemoryPointer,m_nRows,m_nColumns);
    m_buffered.clear();
}

void Matrix::print() const
{
    double *buffer = (double *)malloc(m_nColumns * m_nRows * sizeof(double));
    checkCudaErrors(cudaMemcpy(buffer, m_deviceMemoryPointer, m_nColumns * m_nRows * sizeof(double), cudaMemcpyDeviceToHost));
    std::cout << std::endl;
    for (int i=0; i<m_nRows; i++)
    {
        for (int j=0; j<m_nColumns; j++)
        {
            std::cout << buffer[j+i*m_nColumns] << " ";
        }
        std::cout << std::endl;
    }
    free(buffer);
}

void Matrix::fastGradUpdate(double i_lr, const Vector& i_next, const Vector& i_prev, double i_beta)
{
    CudaDevice::getDevice().cudaMatrixOuterProductUpdate(i_next.deviceMemoryPointer(), i_prev.deviceMemoryPointer(), i_next.size(), i_prev.size(), i_lr, i_beta, m_deviceMemoryPointer);
    m_buffered.clear();
}

void Matrix::fastGradColumnUpdate(double i_lr, const Vector& i_vector, int i_column, double i_beta)
{
    CudaDevice::getDevice().cudaAddVectorToColumn(i_vector.deviceMemoryPointer(),  m_nRows, m_nColumns, i_column, i_lr, i_beta, m_deviceMemoryPointer);
    m_buffered.clear();
}

void Matrix::setZero()
{
    if(m_nColumns + m_nRows < 2)
    {
        throw ESizeIsUnknown();
    }
    CudaDevice::getDevice().setZeroVector(m_deviceMemoryPointer, m_nRows * m_nColumns);
    m_buffered.clear();
}

void Matrix::prepareToSave()
{
    double *buffer = (double *)malloc(m_nColumns * m_nRows * sizeof(double));
    checkCudaErrors(cudaMemcpy(buffer, m_deviceMemoryPointer, m_nRows * m_nColumns * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i=0; i<m_nRows; i++)
    {
        for (int j=0; j<m_nColumns; j++)
        {
            //std::cout << buffer[j+i*m_nColumns] << " ";
            m_buffered.insert(std::pair<std::pair<int,int>, double>(std::pair<int,int>(i,j), buffer[j+i*m_nColumns]));
        }
    }
    free(buffer);

}

void Vector::prepareToSave()
{
    double *buffer = (double *)malloc(m_size * sizeof(double));
    checkCudaErrors(cudaMemcpy(buffer, m_deviceMemoryPointer, m_size * sizeof(double), cudaMemcpyDeviceToHost));
    std::cout << std::endl;
    for (int i=0; i<m_size; i++)
    {
        m_buffered[i] = buffer[i];
    }
    free(buffer);
}

double Matrix::getElement(double i_row, double i_column)
{
    std::pair<int,int> key(i_row, i_column);
    if(m_buffered.find(key) == m_buffered.end())
    {
        throw EIsNotOnHost();
    }
    return m_buffered.at(key);
}

void Vector::update()
{
    double buffer[m_buffered.size()];// = (double*)malloc(m_buffered.size() * sizeof(double));

    for(auto& p : m_buffered)
    {
        buffer[p.first] = p.second;
    }

    setArray(m_buffered.size(), buffer);
}

void Vector::addMatrixColumn(const Matrix& i_rhs, int i_column)
{
    CudaDevice::getDevice().addMatrixColumnToVector(i_rhs.deviceMemoryPointer(), i_rhs.nRows(), i_rhs.nCols(), i_column, m_deviceMemoryPointer);
    m_buffered.clear();
}

void Vector::logisticActivation()
{
    CudaDevice::getDevice().cudaLogisticActivation(m_deviceMemoryPointer, m_size);
    m_buffered.clear();
}
