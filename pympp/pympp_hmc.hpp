#ifndef MPP_PYMPP_HMC_HPP
#define MPP_PYMPP_HMC_HPP

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <mpp/core>


// http://docs.scipy.org/doc/numpy/reference/c-api.dtype.html
// http://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html
template<typename T>
struct numPyTypeTraits;

template<>
struct numPyTypeTraits<float>
{
    static NPY_TYPES getNumPyDType()
    {
        return NPY_FLOAT;
    }
};

template<>
struct numPyTypeTraits<double>
{
    static NPY_TYPES getNumPyDType()
    {
        return NPY_DOUBLE;
    }
};

template<>
struct numPyTypeTraits<long double>
{
    static NPY_TYPES getNumPyDType()
    {
        return NPY_LONGDOUBLE;
    }
};


template<typename _realScalarType>
class pyMppLogPost
{
public:
    typedef _realScalarType realScalarType;
    typedef Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realVectorType;
    typedef Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realDiagMatrixType;
    typedef typename realVectorType::Index indexType;

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

    inline void value(realVectorType   & q, realScalarType & val)
    {
        int nd = 1;
        npy_intp qDims[1]={q.size()};
        PyObject* qIn = PyArray_SimpleNewFromData(nd,qDims,
            numPyTypeTraits<realScalarType>::getNumPyDType(),q.data());

        npy_intp valDims[1]={1};
        PyObject* valOut = PyArray_SimpleNewFromData(nd,valDims,
            numPyTypeTraits<realScalarType>::getNumPyDType(),&val);

        Py_ssize_t tupleSize = 2;
        PyObject* argsTuple = PyTuple_Pack(tupleSize,qIn,valOut);
        PyEval_CallObject(m_p_logPostFunc,argsTuple);
    }

    inline void derivs(realVectorType   & q,realVectorType & dq)
    {
        int nd = 1;
        npy_intp qDims[1]={q.size()};
        PyObject* qIn = PyArray_SimpleNewFromData(nd,qDims,
            numPyTypeTraits<realScalarType>::getNumPyDType(),q.data());

        npy_intp dQDims[1]={dq.size()};
        PyObject* dQOut = PyArray_SimpleNewFromData(nd,dQDims,
            numPyTypeTraits<realScalarType>::getNumPyDType(),dq.data());

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


template<typename T>
Eigen::Map<Eigen::Matrix<T,Eigen::Dynamic,1> > make_vector(T* data,
    size_t const size)
{
    return Eigen::Map<Eigen::Matrix<T,Eigen::Dynamic,1> >(data, size);
}


template<typename realScalarType>
static PyObject* canonicalHamiltonianSamplerImpl(PyObject* self, PyObject* args)
{
    // 1) define the log posterior type
    typedef pyMppLogPost<realScalarType> logPosteriorType;

    // 2) define the canonical Hamiltonian sampler interface
    typedef mpp::canonicalHamiltonianSampler<logPosteriorType> samplerType;
    typedef typename samplerType::realVectorType realVectorType;
    typedef typename samplerType::seedType seedType;

    // 3) get all the arguments from python

    // argument 0 the total number of parameters
    size_t const numParams =
        static_cast<size_t>(PyLong_AsLong(PyTuple_GetItem(args,0)));

    // argument 1 maximum value of epsilon in the leapfrog
    realScalarType const maxEps =
        static_cast<realScalarType>(PyFloat_AsDouble(PyTuple_GetItem(args,1)));

    // argument 2 maximum number of steps in the leapfrog
    size_t const maxNumSteps =
        static_cast<size_t>(PyLong_AsLong(PyTuple_GetItem(args,2)));


    // argument 3 the start point of for the sampler
    PyObject* pyObjStartPoint = PyTuple_GetItem(args, 3);
    PyArrayObject* pyArrObjStartPoint =
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(pyObjStartPoint,
        numPyTypeTraits<realScalarType>::getNumPyDType(), NPY_ARRAY_IN_ARRAY));
    realScalarType* pStartPoint =
        static_cast<realScalarType*>(PyArray_DATA(pyArrObjStartPoint));
    realVectorType  startPoint = make_vector<realScalarType>(pStartPoint,numParams);

    // argument 4 random number seed
    seedType const randSeed =
        static_cast<seedType>(PyLong_AsLong(PyTuple_GetItem(args,4)));

    // argument 5 packet size of the MCMC iteration
    size_t const packetSize =
        static_cast<size_t>(PyLong_AsLong(PyTuple_GetItem(args,5)));

    // argument 6 number samples to be burned
    size_t const numBurn =
        static_cast<size_t>(PyLong_AsLong(PyTuple_GetItem(args,6)));

    // argument 7 number of samples to be taken after burning
    size_t const numSamples =
        static_cast<size_t>(PyLong_AsLong(PyTuple_GetItem(args,7)));

    // argument 8 path to the output files
    std::string const rootPathStr =
        PyString_AsString(PyTuple_GetItem(args, 8));

    // argument 9 print output to the console?
    //http://stackoverflow.com/questions/9316179/what-is-the-correct-way-to-pass-a-boolean-to-a-python-c-extension
    long consoleOutputInt =
        static_cast<long>(PyInt_AsLong(PyTuple_GetItem(args,9)));
    bool const consoleOutput = consoleOutputInt == 0? false : true ;

    // argument 10 delimiter of the chain file
    std::string const delimiter =
        PyString_AsString(PyTuple_GetItem(args, 10));

    // argument 11 precision of the chain file
    unsigned long const precision =
        static_cast<unsigned long>(PyInt_AsLong(PyTuple_GetItem(args,11)));

    // argument 12 diagonal elelements of the M^-1 matrix
    PyObject* pyObjKeDiagMInv = PyTuple_GetItem(args, 12);
    PyArrayObject* pyArrObjKeDiagMInv =
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(pyObjKeDiagMInv,
        numPyTypeTraits<realScalarType>::getNumPyDType(), NPY_ARRAY_IN_ARRAY));
    realScalarType* pKeDiagMInv =
        static_cast<realScalarType*>(PyArray_DATA(pyArrObjKeDiagMInv));
    realVectorType  keDiagMInv = make_vector<realScalarType>(pKeDiagMInv,numParams);

    // argument 13 the log posterior function
    PyObject*   pyObjLogPostFunc  = PyTuple_GetItem(args,13);

    // argument 14 the log posterior derivatives
    PyObject*   pyObjLogPostDerivs  = PyTuple_GetItem(args,14);

    std::cout<<"numParams = "<<numParams<<std::endl;
    std::cout<<"maxEps = "<<maxEps<<std::endl;
    std::cout<<"maxNumSteps = "<<maxNumSteps<<std::endl;

    for(size_t i=0;i<numParams;++i)
    {
        std::cout<<"startPoint["<<i<<"] = "<<startPoint(i)<<std::endl;
    }
    std::cout<<"randSeed = "<<randSeed<<std::endl;
    std::cout<<"packetSize = "<<packetSize<<std::endl;
    std::cout<<"numBurn = "<<numBurn<<std::endl;
    std::cout<<"numSamples = "<<numSamples<<std::endl;
    std::cout<<"rootPathStr = "<<rootPathStr<<std::endl;
    std::cout<<"consoleOutput = "<<consoleOutput<<std::endl;
    std::cout<<"delimiter = "<<delimiter<<std::endl;
    std::cout<<"precision = "<<precision<<std::endl;
    for(size_t i=0;i<numParams;++i)
    {
        std::cout<<"keDiagMInv["<<i<<"] = "<<keDiagMInv(i)<<std::endl;
    }

    // 4) make the posterior distribution
    logPosteriorType G(pyObjLogPostFunc,pyObjLogPostDerivs,numParams);


    // 5) define the sampler
    samplerType canonHamiltSampler(
        G,
        numParams,
        maxEps,
        maxNumSteps,
        startPoint,
        randSeed,
        packetSize,
        numBurn,
        numSamples,
        rootPathStr,
        consoleOutput,
        delimiter,
        precision,
        keDiagMInv
    );

    // 6) finally run the sampler
    canonHamiltSampler.run();

    return Py_None;
}


#endif //MPP_PYMPP_HMC_HPP
