#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <string>
//#include "pympp_hmc.hpp"
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

static PyObject * MPPError;

template<typename realScalarType>
static PyObject* canonicalHamiltonianSamplerImpl(PyObject* self, PyObject* args)
{
    // 2) get all the arguments from python
    PyObject*   pyObjLogPostFunc  = PyTuple_GetItem(args,0);

    size_t const numParams =
        static_cast<size_t>(PyLong_AsSsize_t(PyTuple_GetItem(args,1)));

    realScalarType const maxEps =
        static_cast<realScalarType>(PyFloat_AsDouble(PyTuple_GetItem(args,2)));

    size_t const maxNumSteps =
        static_cast<size_t>(PyLong_AsSsize_t(PyTuple_GetItem(args,3)));

    PyObject* pyObjStartPoint = PyTuple_GetItem(args, 4);
    PyArrayObject* pyArrObjStartPoint =
        reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(pyObjStartPoint,
        numPyTypeTraits<realScalarType>::getNumPyDType(), NPY_ARRAY_IN_ARRAY));
    realScalarType* startPoint =
        static_cast<realScalarType*>(PyArray_DATA(pyArrObjStartPoint));

    size_t const randSeed =
        static_cast<size_t>(PyLong_AsSsize_t(PyTuple_GetItem(args,5)));

    size_t const packetSize =
        static_cast<size_t>(PyLong_AsSsize_t(PyTuple_GetItem(args,6)));

    size_t const numBurn =
        static_cast<size_t>(PyLong_AsSsize_t(PyTuple_GetItem(args,7)));

    size_t const numSamples =
        static_cast<size_t>(PyLong_AsSsize_t(PyTuple_GetItem(args,8)));

    std::string const & rootPathStr =
        PyString_AsString(PyTuple_GetItem(args, 9));

    //bool const consoleOutput = if(PyTuple_GetItem(args, 9) == 


    return Py_None;
}

static PyObject* canonicalHamiltonianSampler(PyObject* self, PyObject* args)
{
    //http://docs.scipy.org/doc/numpy/reference/c-api.array.html
    Py_ssize_t tupleSize = PyTuple_Size(args);

    if(tupleSize != 1)
    {
        PyErr_SetString(PyExc_RuntimeError, "Unexpected number of arguments");
        return NULL;
    }

    std::cout<<"number of args = "<<tupleSize<<std::endl;

    // 1) determine the floating point type from start point

    PyArrayObject* pyArrObjStartPoint =
        (PyArrayObject*) (PyTuple_GetItem(args, 4));

    // PyArray_DTYPE new in version 1.7.
    PyArray_Descr * npArrTypeInfo = PyArray_DTYPE(pyArrObjStartPoint);

    if(npArrTypeInfo->type == 'f')
    {
        return canonicalHamiltonianSamplerImpl<float>(self,args);
    }
    else if(npArrTypeInfo->type == 'd')
    {
        return canonicalHamiltonianSamplerImpl<double>(self,args);
    }
    else if(npArrTypeInfo->type == 'g')
    {
        return canonicalHamiltonianSamplerImpl<long double>(self,args);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "unknown type in argument 0. Was expecting numpy floating point array");
        return NULL;
    }


    return Py_None;
}

static PyMethodDef MPPMethods[] =
{
    // Python Name, C Name, Arguments, Description
    {"canonicalHamiltonianSampler", canonicalHamiltonianSampler,
    METH_VARARGS, "Python Interface for MPP Canonical Hamiltonian Sampler"},
    {NULL,NULL,0,NULL}
};

PyMODINIT_FUNC initmpp_module()
{
    PyObject *m;
    m = Py_InitModule("mpp_module",MPPMethods);
    if(m==NULL)
    {
        return;
    }

    MPPError = PyErr_NewException((char*)std::string("mpp_module.error").c_str(), NULL, NULL);
    Py_INCREF(MPPError);
    PyModule_AddObject(m, "error", MPPError);
    import_array();
}

int main(int argc, char *argv[])
{
    Py_SetProgramName(argv[0]);
    Py_Initialize();
    initmpp_module();
    return 0;
}
