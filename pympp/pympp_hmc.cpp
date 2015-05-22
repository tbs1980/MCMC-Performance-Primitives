#define BOOST_ALL_DYN_LINK
#include "pympp_hmc.hpp"

static PyObject * MPPError;

static PyObject* canonicalHamiltonianSampler(PyObject* self, PyObject* args)
{
    //http://docs.scipy.org/doc/numpy/reference/c-api.array.html
    Py_ssize_t tupleSize = PyTuple_Size(args);

    if(tupleSize != 15)
    {
        PyErr_SetString(PyExc_RuntimeError, "Unexpected number of arguments");
        return NULL;
    }

    std::cout<<"number of args = "<<tupleSize<<std::endl;

    // 1) determine the floating point type from start point

    PyArrayObject* pyArrObjStartPoint =
        (PyArrayObject*) (PyTuple_GetItem(args, 3));

    // PyArray_DTYPE new in version 1.7.
    PyArray_Descr * npArrTypeInfo = PyArray_DTYPE(pyArrObjStartPoint);

    if(npArrTypeInfo->type == 'f')
    {
        std::cout<<"We are working in float "<<std::endl;
        return canonicalHamiltonianSamplerImpl<float>(self,args);
    }
    else if(npArrTypeInfo->type == 'd')
    {
        std::cout<<"We are working in double "<<std::endl;
        return canonicalHamiltonianSamplerImpl<double>(self,args);
    }
    else if(npArrTypeInfo->type == 'g')
    {
        std::cout<<"We are working in long double "<<std::endl;
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
