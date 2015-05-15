#ifndef MPP_PYMPP_HMC_HPP
#define MPP_PYMPP_HMC_HPP

#include <Python.h>
#include <numpy/arrayobject.h>


template<typename _realScalarType>
class pyMppLogPost
{
public:
    typedef _realScalarType realScalarType;
    typedef Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realVectorType;
    typedef Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realDiagMatrixType;
    typedef typename realVectorType::Index indexType;

    pyMppLogPost(
        PyObject* p_log_post_func,
        PyObject* p_log_post_derivs,
        indexType const numDims
        )
    {
        m_p_log_post_func = p_log_post_func;
        m_p_log_post_derivs = p_log_post_derivs;
        m_numDims = numDims;
    }

    inline void value(realVectorType  const & q, realScalarType & val) const
    {
        int nd = 1;
        npy_intp dims[1]={q.size()};
        PyObject* qin = PyArray_SimpleNewFromData(nd,dims,NPY_DOUBLE,q.data());
        Py_ssize_t tuple_size = 1;
        PyObject* tuple = PyTuple_Pack(tuple_size,qin);
        PyObject* outputs = PyArray_FROM_OTF(PyEval_CallObject(m_p_log_post_func,args_tuple),NPY_DOUBLE,NPY_ARRAY_IN_ARRAY);
    }

    inline void derivs(realVectorType  const & q,realVectorType & dq) const
    {
    }

    inline indexType numDims(void) const
    {
        return m_numDims;
    }

private:
    PyObject* m_p_log_post_func;
    PyObject* m_p_log_post_derivs;
    indexType m_numDims;
};


#endif //MPP_PYMPP_HMC_HPP
