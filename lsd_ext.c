#include <Python.h>
#include <numpy/arrayobject.h>
#include "lsd_1.6/lsd.h"

PyDoc_STRVAR(lsd__doc__,
"LSD Simple Interface.\n\n"
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

static PyObject* _lsd(PyObject* self, PyObject* args)
{
    PyObject *image;
    int n_out;
    double *output;
    npy_intp dims[2], *shape;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &image)) {
        return NULL;
    }
    if (PyArray_NDIM(image) != 2) {
        PyErr_SetString(PyExc_ValueError, "only gray image accepted");
        return NULL;
    }
    image = PyArray_FROM_OTF(image, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    shape = PyArray_DIMS(image);
    output = lsd(&n_out, (double*)PyArray_DATA(image), (int)shape[0], (int)shape[1]);
    dims[0] = n_out, dims[1] = 7;
    Py_DECREF(image);
    return PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, (void*)output);
}

PyDoc_STRVAR(lsd_scale__doc__,
"LSD Simple Interface with Scale.\n\n"
"@param img         Input image data. It must be an array of size X x Y.\n\n"
"@param scale       When different from 1.0, LSD will scale the input image\n"
"                   by 'scale' factor by Gaussian filtering, before detecting\n"
"                   line segments.\n"
"                   Example: if scale=0.8, the input image will be subsampled\n"
"                   to 80% of its size, before the line segment detector\n"
"                   is applied.\n"
"                   Suggested value: 0.8\n\n"
"@return            Return a double array of size 7 x n_out, containing the\n"
"                   list of line segments detected.\n"
"                   The seven values are:\n"
"                   - x1,y1,x2,y2,width,p,-log10(NFA)\n"
"                   .\n"
"                   for a line segment from coordinates (x1,y1) to (x2,y2),\n"
"                   a width 'width', an angle precision of p in (0,1) given\n"
"                   by angle_tolerance/180 degree, and NFA value 'NFA'.\n"
);

static PyObject* _lsd_scale(PyObject* self, PyObject* args)
{
    PyObject *image;
    int n_out;
    double *output, scale;
    npy_intp dims[2], *shape;

    if (!PyArg_ParseTuple(args, "O!d", &PyArray_Type, &image, &scale)) {
        return NULL;
    }
    if (PyArray_NDIM(image) != 2) {
        PyErr_SetString(PyExc_ValueError, "only gray image accepted");
        return NULL;
    }
    image = PyArray_FROM_OTF(image, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    shape = PyArray_DIMS(image);
    output = lsd_scale(&n_out, (double*)PyArray_DATA(image), (int)shape[0], (int)shape[1], scale);
    dims[0] = n_out, dims[1] = 7;
    Py_DECREF(image);
    return PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, (void*)output);
}

PyDoc_STRVAR(lsd_scale_region__doc__,
"LSD Simple Interface with Scale and Region output.\n\n"
"@param img         Input image data. It must be an array of size X x Y.\n\n"
"@param scale       When different from 1.0, LSD will scale the input image\n"
"                   by 'scale' factor by Gaussian filtering, before detecting\n"
"                   line segments.\n"
"                   Example: if scale=0.8, the input image will be subsampled\n"
"                   to 80% of its size, before the line segment detector\n"
"                   is applied.\n"
"                   Suggested value: 0.8\n\n"
"@param reg_img     Optional output: if desired, LSD will return an\n"
"                   int image where each pixel indicates the line segment\n"
"                   to which it belongs. Unused pixels have the value '0',\n"
"                   while the used ones have the number of the line segment,\n"
"                   numbered 1,2,3,..., in the same order as in the\n"
"                   output list. If desired, a non NULL int** pointer must\n"
"                   be assigned, and LSD will make that the pointer point\n"
"                   to an int array of size reg_x x reg_y, where the pixel\n"
"                   value at (x,y) is obtained with (*reg_img)[x+y*reg_x].\n"
"                   Note that the resulting image has the size of the image\n"
"                   used for the processing, that is, the size of the input\n"
"                   image scaled by the given factor 'scale'. If scale!=1\n"
"                   this size differs from XxY and that is the reason why\n"
"                   its value is given by reg_x and reg_y.\n"
"                   Suggested value: NULL\n\n"
"@return            Return a double array of size 7 x n_out, containing the\n"
"                   list of line segments detected.\n"
"                   The seven values are:\n"
"                   - x1,y1,x2,y2,width,p,-log10(NFA)\n"
"                   .\n"
"                   for a line segment from coordinates (x1,y1) to (x2,y2),\n"
"                   a width 'width', an angle precision of p in (0,1) given\n"
"                   by angle_tolerance/180 degree, and NFA value 'NFA'.\n"
);

static PyObject* _lsd_scale_region(PyObject* self, PyObject* args)
{
    PyObject *image_arr, *region_arr, *output_arr;
    int n_out, *reg_img, reg_x, reg_y;
    double *output, scale;
    npy_intp dims[2], *shape, reg_dims[2];

    if (!PyArg_ParseTuple(args, "O!dO!", &PyArray_Type, &image_arr, &scale, &PyArray_Type, &region_arr)) {
        return NULL;
    }
    if (PyArray_NDIM(image_arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "only gray image accepted");
        return NULL;
    }
    image_arr = PyArray_FROM_OTF(image_arr, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    shape = PyArray_DIMS(image_arr);
    output = lsd_scale_region(&n_out, (double*)PyArray_DATA(image_arr), (int)shape[0], (int)shape[1], scale, &reg_img, &reg_x, &reg_y);
    dims[0] = n_out, dims[1] = 7;
    output_arr = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, (void*)output);
    reg_dims[0] = reg_x, reg_dims[1] = reg_y;
    PyArray_Dims _reg_dims = {reg_dims, 2};
    PyArray_Resize((PyArrayObject*)region_arr, &_reg_dims, 0, NPY_FORTRANORDER);
    PyArray_CopyObject((PyArrayObject*)region_arr, PyArray_SimpleNewFromData(2, reg_dims, NPY_INT, (void*)reg_img));
    Py_DECREF(image_arr);
    free(output);
    return output_arr;
}

PyDoc_STRVAR(LineSegmentDetection__doc__,
"LSD Full Interface\n\n"
"@param img         Input image data. It must be an array of size X x Y.\n\n"
"@param scale       When different from 1.0, LSD will scale the input image\n"
"                   by 'scale' factor by Gaussian filtering, before detecting\n"
"                   line segments.\n"
"                   Example: if scale=0.8, the input image will be subsampled\n"
"                   to 80% of its size, before the line segment detector\n"
"                   is applied.\n"
"                   Suggested value: 0.8\n\n"
"@param sigma_scale When scale!=1.0, the sigma of the Gaussian filter is:\n"
"                   sigma = sigma_scale / scale,   if scale <  1.0\n"
"                   sigma = sigma_scale,           if scale >= 1.0\n"
"                   Suggested value: 0.6\n\n"
"@param quant       Bound to the quantization error on the gradient norm.\n"
"                   Example: if gray levels are quantized to integer steps,\n"
"                   the gradient (computed by finite differences) error\n"
"                   due to quantization will be bounded by 2.0, as the\n"
"                   worst case is when the error are 1 and -1, that\n"
"                   gives an error of 2.0.\n"
"                   Suggested value: 2.0\n\n"
"@param ang_th      Gradient angle tolerance in the region growing\n"
"                   algorithm, in degrees.\n"
"                   Suggested value: 22.5\n\n"
"@param log_eps     Detection threshold, accept if -log10(NFA) > log_eps.\n"
"                   The larger the value, the more strict the detector is,\n"
"                   and will result in less detections.\n"
"                   (Note that the 'minus sign' makes that this\n"
"                   behavior is opposite to the one of NFA.)\n"
"                   The value -log10(NFA) is equivalent but more\n"
"                   intuitive than NFA:\n"
"                   - -1.0 gives an average of 10 false detections on noise\n"
"                   -  0.0 gives an average of 1 false detections on noise\n"
"                   -  1.0 gives an average of 0.1 false detections on nose\n"
"                   -  2.0 gives an average of 0.01 false detections on noise\n"
"                   .\n"
"                   Suggested value: 0.0\n\n"
"@param density_th  Minimal proportion of 'supporting' points in a rectangle.\n"
"                   Suggested value: 0.7\n\n"
"@param n_bins      Number of bins used in the pseudo-ordering of gradient\n"
"                   modulus.\n"
"                   Suggested value: 1024\n\n"
"@param reg_img     Optional output: if desired, LSD will return an\n"
"                   int image where each pixel indicates the line segment\n"
"                   to which it belongs. Unused pixels have the value '0',\n"
"                   while the used ones have the number of the line segment,\n"
"                   numbered 1,2,3,..., in the same order as in the\n"
"                   output list. If desired, a non NULL int** pointer must\n"
"                   be assigned, and LSD will make that the pointer point\n"
"                   to an int array of size reg_x x reg_y, where the pixel\n"
"                   value at (x,y) is obtained with (*reg_img)[x+y*reg_x].\n"
"                   Note that the resulting image has the size of the image\n"
"                   used for the processing, that is, the size of the input\n"
"                   image scaled by the given factor 'scale'. If scale!=1\n"
"                   this size differs from XxY and that is the reason why\n"
"                   its value is given by reg_x and reg_y.\n"
"                   Suggested value: NULL\n\n"
"@return            Return a double array of size 7 x n_out, containing the\n"
"                   list of line segments detected.\n"
"                   The seven values are:\n"
"                   - x1,y1,x2,y2,width,p,-log10(NFA)\n"
"                   .\n"
"                   for a line segment from coordinates (x1,y1) to (x2,y2),\n"
"                   a width 'width', an angle precision of p in (0,1) given\n"
"                   by angle_tolerance/180 degree, and NFA value 'NFA'.\n"
);

static PyObject* _LineSegmentDetection(PyObject* self, PyObject* args)
{
    PyObject *image_arr, *region_arr, *output_arr;
    int n_out, *reg_img, reg_x, reg_y, n_bins;
    double *output, scale, sigma_scale, quant, ang_th, log_eps, density_th;
    npy_intp dims[2], *shape, reg_dims[2];

    if (!PyArg_ParseTuple(args, "O!ddddddiO!", &PyArray_Type, &image_arr, &scale, &sigma_scale, &quant, &ang_th, &log_eps, &density_th, &n_bins, &PyArray_Type, &region_arr)) {
        return NULL;
    }
    if (PyArray_NDIM(image_arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "only gray image accepted");
        return NULL;
    }
    image_arr = PyArray_FROM_OTF(image_arr, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    shape = PyArray_DIMS(image_arr);
    output = LineSegmentDetection(&n_out, (double*)PyArray_DATA(image_arr), (int)shape[0], (int)shape[1], scale, sigma_scale, quant, ang_th, log_eps, density_th, n_bins, &reg_img, &reg_x, &reg_y);
    dims[0] = n_out, dims[1] = 7;
    output_arr = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, (void*)output);
    reg_dims[0] = reg_x, reg_dims[1] = reg_y;
    PyArray_Dims _reg_dims = {reg_dims, 2};
    PyArray_Resize((PyArrayObject*)region_arr, &_reg_dims, 0, NPY_FORTRANORDER);
    PyArray_CopyObject((PyArrayObject*)region_arr, PyArray_SimpleNewFromData(2, reg_dims, NPY_INT, (void*)reg_img));
    Py_DECREF(image_arr);
    free(output);
    return output_arr;
}

static PyMethodDef __methods[] = {
    {"lsd", (PyCFunction)_lsd, METH_VARARGS, lsd__doc__},
    {"lsd_scale", (PyCFunction)_lsd_scale, METH_VARARGS, lsd_scale__doc__},
    {"lsd_scale_region", (PyCFunction)_lsd_scale_region, METH_VARARGS, lsd_scale_region__doc__},
    {"LineSegmentDetection", (PyCFunction)_LineSegmentDetection, METH_VARARGS, LineSegmentDetection__doc__},
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
    import_array();
    return module;
}
