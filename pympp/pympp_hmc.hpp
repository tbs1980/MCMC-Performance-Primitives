#ifndef MPP_PYMPP_HMC_HPP
#define MPP_PYMPP_HMC_HPP


// http://docs.scipy.org/doc/numpy/reference/c-api.dtype.html
template<typename T>
struct numPyTypeTraits;

template<>
struct numPyTypeTraits<float>
{
    typedef typename NPY_FLOAT realScalarType;
};

template<>
struct numPyTypeTraits<float>
{
    typedef typename NPY_FLOAT32 realScalarType;
};

template<>
struct numPyTypeTraits<double>
{
    typedef typename NPY_FLOAT64 realScalarType;
};

template<>
struct numPyTypeTraits<double>
{
    typedef typename NPY_DOUBLE realScalarType;
};

template<>
struct numPyTypeTraits<long double>
{
    typedef typename NPY_LONGDOUBLE realScalarType;
};

template<typename _realScalarType>
class pyMppLogPost
{
public:
    typedef _realScalarType realScalarType;
    typedef Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realVectorType;
    typedef Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realDiagMatrixType;
    typedef typename realVectorType::Index indexType;

    typedef typename numPyTypeTraits<realScalarType>::realScalarType numPyRealScalarType

    pyMppLogPost(
        PyObject* p_logPostFunc,
        PyObject* p_logPostDerivs,
        indexType const numDims
        )
    {
        m_p_logPostFunc = p_logPostFunc;
        m_p_logPostDerivs = p_logPostDerivs;
        m_numDims = numDims;
    }

    inline void value(realVectorType  const & q, realScalarType & val) const
    {
        int nd = 1;
        npy_intp qDims[1]={q.size()};
        PyObject* qIn = PyArray_SimpleNewFromData(nd,qDims,numPyRealScalarType,q.data());

        npy_intp valDims[1]={1};
        PyObject* valOut = PyArray_SimpleNewFromData(nd,valDims,numPyRealScalarType,val);

        Py_ssize_t tupleSize = 2;
        PyObject* argsTuple = PyTuple_Pack(tupleSize,qIn,valOut);
        PyEval_CallObject(m_p_logPostFunc,argsTuple);
    }

    inline void derivs(realVectorType  const & q,realVectorType & dq) const
    {
        int nd = 1;
        npy_intp qDims[1]={q.size()};
        PyObject* qIn = PyArray_SimpleNewFromData(nd,qDims,numPyRealScalarType,q.data());

        npy_intp dQDims[1]={dq.size()};
        PyObject* dQOut = PyArray_SimpleNewFromData(nd,dQDims,numPyRealScalarType,dQOut);

        Py_ssize_t tupleSize = 2;
        PyObject* argsTuple = PyTuple_Pack(tupleSize,qIn,dQOut);
        PyEval_CallObject(m_p_logPostDerivs,argsTuple);
    }

    inline indexType numDims(void) const
    {
        return m_numDims;
    }

private:
    PyObject* m_p_logPostFunc;
    PyObject* m_p_logPostDerivs;
    indexType m_numDims;
};


#endif //MPP_PYMPP_HMC_HPP
