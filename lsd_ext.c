#include <Python.h>
#include <numpy/arrayobject.h>
#include "lsd_1.6/lsd.h"


static PyObject* _lsd(PyObject* self, PyObject* args)
{
    PyArrayObject* image;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &image)) {
        return NULL;
    }
    if (PyArray_NDIM(image) != 2) {
        PyErr_SetString(PyExc_ValueError, "only gray image accepted");
        return NULL;
    }
    image = PyArray_FromArray(image, PyArray_DescrFromType(NPY_DOUBLE), NPY_ARRAY_F_CONTIGUOUS);
    npy_intp* shape = PyArray_SHAPE(image);
    int n_out;
    double* output = lsd(&n_out, PyArray_DATA(image), shape[0], shape[1]);
    npy_intp dims[2] = {n_out, 7};
    return PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, (void*)output);
}

PyDoc_STRVAR(lsd__doc__,
"LSD Simple Interface\n\n"
"@param img         Input image data. It must be an array of size X x Y.\n\n"
"@return            Return a double array of size 7 x n_out, containing the\n"
"                   list of line segments detected.\n"
"                   The seven values are:\n"
"                   - x1,y1,x2,y2,width,p,-log10(NFA)\n"
"                   .\n"
"                   for a line segment from coordinates (x1,y1) to (x2,y2),\n"
"                   a width 'width', an angle precision of p in (0,1) given\n"
"                   by angle_tolerance/180 degree, and NFA value 'NFA'.\n"
);

static PyMethodDef __methods[] = {
    {"lsd", (PyCFunction)_lsd, METH_VARARGS, lsd__doc__},
    {NULL, NULL, 0, NULL}    /* sentinel */
};

static PyModuleDef __module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "lsd_ext",
    .m_doc = "C extension of Python for LSD (von Gioi et al, 2010)",
    .m_size = -1,
    .m_methods = __methods
};

PyMODINIT_FUNC PyInit_lsd_ext(void)
{
    PyObject* module = PyModule_Create(&__module);
    if (module == NULL) {
        return NULL;
    }
    import_array();
    if (PyErr_Occurred()) {
        return NULL;
    }
    return module;
}