#include <map>
#include "cuda-backend.h"

class Vector;
class Matrix;
class Layer;

enum {MUL, SUM};

template<typename T>
class CudaProxyLog
{
public:
    CudaProxyLog(T *i_devicePointer):
        m_deviceMemoryPointer(i_devicePointer) {}
    T *deviceMemoryPointer() const {return m_deviceMemoryPointer;}
private:
    T *m_deviceMemoryPointer;
};

template<typename T>
class CudaValue
{
public:
    CudaValue(T i_value)
    {
        cudaSetDevice(1);
        checkCudaErrors(cudaMalloc((void **) &m_deviceMemoryPointer, sizeof(T)));
        checkCudaErrors(cudaMemcpy(m_deviceMemoryPointer, &i_value,  sizeof(T), cudaMemcpyHostToDevice));
    }

    CudaValue<T>& operator=(const T& i_value)
    {
        checkCudaErrors(cudaMemcpy(m_deviceMemoryPointer, &i_value,  sizeof(T), cudaMemcpyHostToDevice));
        return *this;
    }

    CudaValue<T>& operator=(const CudaValue<T>& i_value)
    {
        checkCudaErrors(cudaMemcpy(m_deviceMemoryPointer, i_value.deviceMemoryPointer(), sizeof(T), cudaMemcpyDeviceToDevice));
        return *this;
    }

    CudaValue<T>& operator+=(const CudaValue<T>& i_value)
    {
        CudaDevice::getDevice().cudaAddScalarToScalar(i_value.deviceMemoryPointer(), m_deviceMemoryPointer);
        return *this;
    }

    CudaValue<T>& operator+=(const CudaProxyLog<T>& i_value)
    {
        CudaDevice::getDevice().cudaAddLog(i_value.deviceMemoryPointer(), m_deviceMemoryPointer);
        return *this;
    }

    ~CudaValue()
    {
        checkCudaErrors(cudaFree(m_deviceMemoryPointer));
    }

    T *deviceMemoryPointer() const {return m_deviceMemoryPointer;}

    explicit operator T()
    {
        T o_result = 0;
        checkCudaErrors(cudaMemcpy(&o_result, m_deviceMemoryPointer, sizeof(T), cudaMemcpyDeviceToHost));
        return o_result;
    }

private:
    T *m_deviceMemoryPointer;
};

template<typename LeftT, typename RightT, size_t>
struct ProxyOp
{
    const LeftT& m_leftOperand;
    const RightT& m_rightOperand;

    ProxyOp(const LeftT& i_leftOperand, const RightT& i_rightOperand) : m_leftOperand(i_leftOperand), m_rightOperand(i_rightOperand){}
    const LeftT& leftOperand() const {return m_leftOperand;}
    const RightT& rightOperand() const {return m_rightOperand;}
};

class Matrix
{
public:
    Matrix() : m_transposed(false), m_nColumns(-1), m_nRows(-1) {}
    ~Matrix() {cudaFree(m_deviceMemoryPointer);}
    void setZero(size_t i_nRows, size_t i_nColumns);
    void setZero();
    void setMatrix(double *i_initializer, size_t i_nRows, size_t i_nColumns);
    //ProxyMatrixColumn column(int i_column1);
    void print() const;
    double *deviceMemoryPointer() const {return m_deviceMemoryPointer;}
    int nCols() const {return m_nColumns;}
    int nRows() const {return m_nRows;}
    const Matrix& transpose()
    {
        m_transposed = !m_transposed;
        return *this;
    }
    bool transposed() const {return m_transposed;}
    void fastGradUpdate(double i_lr, const Vector& i_next, const Vector& i_prev, double i_beta);
    void fastGradColumnUpdate(double i_lr, const Vector& i_next, int i_column, double i_beta);
    void addExpression(const Matrix& i_rhs, double i_beta)
    {
        CudaDevice::getDevice().addMatrixToMatrixAndScale(i_rhs.deviceMemoryPointer(), m_nRows, m_nColumns, i_beta, m_deviceMemoryPointer);
        m_buffered.clear();
    }
    void addColumnToColumn(int i_targetColumn, const Matrix& i_update, int i_updateColumn, double i_beta)
    {
        CudaDevice::getDevice().cudaAddColumnToColumn(i_update.deviceMemoryPointer(), m_nRows, m_nColumns, i_updateColumn, m_deviceMemoryPointer, i_targetColumn, i_beta);
        m_buffered.clear();
    }
    void setZeroColumn(int i_column)
    {
        CudaDevice::getDevice().setZeroColumn(m_deviceMemoryPointer, m_nRows, m_nColumns, i_column);
        m_buffered.clear();
    }
    double getElement(double i_row, double i_column);
    void prepareToSave();
    Matrix(const Matrix& rhs)
    {
        CudaDevice::getDevice().copy(m_deviceMemoryPointer, rhs.deviceMemoryPointer(), m_nColumns * m_nRows);
        m_nColumns = rhs.nCols();
        m_nRows = rhs.nRows();
        m_buffered.clear();
    }

    void operator=(const Matrix& rhs)
    {
        CudaDevice::getDevice().copy(rhs.deviceMemoryPointer(), m_deviceMemoryPointer, m_nColumns * m_nRows);
        m_nColumns = rhs.nCols();
        m_nRows = rhs.nRows();
        m_buffered.clear();
    }

private:
    bool m_transposed;
    double *m_deviceMemoryPointer;
    int m_nColumns, m_nRows;
    std::map<std::pair<int,int>, double> m_buffered;
};

class Vector
{
public:
    Vector() {m_size = -1; m_deviceMemoryPointer = NULL; }
    ~Vector()
    {
     //   std::cout << "DESTRUCT!!!!!" <<std::endl;
        if(m_deviceMemoryPointer != NULL) cudaFree(m_deviceMemoryPointer);
      //  printf("%llu \n", (size_t)m_deviceMemoryPointer);
    }
    void setConstant(size_t i_size, double i_constToFill);
    void setConstant(double i_constToFill);
    void setZero() {CudaDevice::getDevice().setZeroVector(m_deviceMemoryPointer, m_size);}
    void setArray(size_t i_size, double *i_actiivationInitializer);
    void copyActivation(const Layer& rhs);
    int size() const {return m_size;}
    void softmaxActivation()
    {
        CudaDevice::getDevice().cudaSoftmaxActivation(m_deviceMemoryPointer, m_size);
        m_buffered.clear();
    }
    void operator=(const ProxyOp<Matrix, Vector, MUL>& i_proxyExpr)
    {
        CudaDevice::getDevice().cudaVectorByMatrix(i_proxyExpr.leftOperand().deviceMemoryPointer(),
                                                   i_proxyExpr.rightOperand().deviceMemoryPointer(),
                                                   i_proxyExpr.leftOperand().nRows(),
                                                   i_proxyExpr.leftOperand().nCols(),
                                                   i_proxyExpr.leftOperand().transposed(),
                                                   m_deviceMemoryPointer);
        m_buffered.clear();
    }

    void operator+=(const ProxyOp<Matrix, Vector, MUL>& i_proxyExpr)
    {
        CudaDevice::getDevice().cudaVectorByMatrix(i_proxyExpr.leftOperand().deviceMemoryPointer(),
                                                   i_proxyExpr.rightOperand().deviceMemoryPointer(),
                                                   i_proxyExpr.leftOperand().nRows(),
                                                   i_proxyExpr.leftOperand().nCols(),
                                                   i_proxyExpr.leftOperand().transposed(),
                                                   m_deviceMemoryPointer, true);
        m_buffered.clear();
    }

    void operator=(const ProxyOp<Vector, Vector, SUM>& i_proxyExpr)
    {
        CudaDevice::getDevice().cudaVectorSum(i_proxyExpr.leftOperand().deviceMemoryPointer(),
                                                   i_proxyExpr.rightOperand().deviceMemoryPointer(),
                                                   i_proxyExpr.leftOperand().size(),
                                                   m_deviceMemoryPointer);
        m_buffered.clear();
    }

    Vector(Vector&& rhs)
    {
        m_deviceMemoryPointer = rhs.deviceMemoryPointer();
        m_size = rhs.size();
        rhs.erase();
        m_buffered.clear();
    }

    void operator=(Vector&& rhs)
    {
        m_deviceMemoryPointer = rhs.deviceMemoryPointer();
        m_size = rhs.size();
        rhs.erase();
        m_buffered.clear();
    }

    Vector(const Vector& rhs)
    {
        CudaDevice::getDevice().copy(m_deviceMemoryPointer, rhs.deviceMemoryPointer(), m_size);
        m_size = rhs.size();
        m_buffered.clear();
    }

    void operator=(const Vector& rhs)
    {
        CudaDevice::getDevice().copy(rhs.deviceMemoryPointer(), m_deviceMemoryPointer, m_size);
        m_size = rhs.size();
        m_buffered.clear();
    }

    double& operator[](int i_coordinate)
    {
        if(m_buffered.find(i_coordinate)==m_buffered.end())
        {
            m_buffered[i_coordinate] = getCoordinate_(i_coordinate);
        }
        return m_buffered[i_coordinate];
    }

    CudaProxyLog<double> elementLog(int i_coordinate)
    {
        return CudaProxyLog<double>(m_deviceMemoryPointer + i_coordinate);
    }

    void print() const;
    double *deviceMemoryPointer() const {return m_deviceMemoryPointer;}
    void erase()
    {
        m_deviceMemoryPointer = 0;
        m_size = -1;
        m_buffered.clear();
    }

    void addMatrixColumn(const Matrix& i_rhs, int i_column);
    void logisticActivation();
    void prepareToSave();
    void update();

private:
    double getCoordinate_(int i_coordinate)
    {
        return CudaDevice::getDevice().cudaGetVectorCoordinate(m_deviceMemoryPointer, i_coordinate);
    }
    double *m_deviceMemoryPointer;
    int m_size;
    std::map<int,double> m_buffered;
};

class Layer
{
public:
    Vector ac;
    Vector er;
    void setConstant(size_t i_size, double i_constToFill);
    void setArray(size_t i_size, double *i_actiivationInitializer);
    int size() const {return m_size;}
    void fastOutputError(int i_trueWord)
    {
        CudaDevice::getDevice().cudaOutputErrorCompute(ac.deviceMemoryPointer(),m_size, i_trueWord, er.deviceMemoryPointer());
    }

    void logisticErrorActivation()
    {
        CudaDevice::getDevice().logisticErrorActivation(ac.deviceMemoryPointer(), er.deviceMemoryPointer(), m_size);
    }

    Layer() = default;

    Layer(Layer&& rhs)
    {
        ac = std::move(rhs.ac);
        er = std::move(rhs.er);
        m_size = rhs.size();
    }

    Layer(Layer& rhs)
    {
        ac = rhs.ac;
        er = rhs.er;
        m_size = rhs.size();
    }

    void operator =(Layer& rhs)
    {
        ac = rhs.ac;
        er = rhs.er;
        m_size = rhs.size();
    }

private:
    int m_size;
};

inline ProxyOp<Matrix, Vector,  MUL> operator *(const Matrix& i_mat, const Vector& i_vec)
{
    return ProxyOp<Matrix, Vector,  MUL>(i_mat, i_vec);
}

inline ProxyOp<Vector, Vector,  SUM> operator +(const Vector& i_left, const Vector& i_right)
{
    return ProxyOp<Vector, Vector,  SUM>(i_left, i_right);
}

//template<typename LeftT, typename RightT, size_t OpCode>
//ProxyOp<ProxyOp<LeftT, RightT, OpCode>, ProxyMatrixColumn, SUM> operator +(const ProxyOp<LeftT, RightT, OpCode>& i_proxyOp, const ProxyMatrixColumn& i_proxyMatrixBlock)
//{

//}


class ESizeIsUnknown
{
public:
    ESizeIsUnknown() {std::cout << "Matrix size is unknown at line" << __LINE__ << std::endl;}
};

class EIsNotOnHost
{
public:
    EIsNotOnHost() {std::cout << "The key is not buffered. Either matrix is not copied to the host or there is no such element." << __LINE__ << std::endl;}
};
