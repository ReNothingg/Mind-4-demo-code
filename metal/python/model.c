#include <Python.h>

#include <mind-four.h>

#include "module.h"

static int PymindfourModel_init(PymindfourModel *self, PyObject *args, PyObject *kwargs)
{
    enum mindfour_status status;
    const char *filepath;

    if (!PyArg_ParseTuple(args, "s", &filepath))
    {
        return -1;
    }
    status = mindfour_model_create_from_file(filepath, &self->handle);
    if (status != mindfour_status_success)
    {

        return -1;
    }
    return 0;
}

static void PymindfourModel_dealloc(PymindfourModel *self)
{
    (void)mindfour_model_release(self->handle);
    self->handle = NULL;
    PyObject_Del((PyObject *)self);
}

static PyObject *PymindfourModel_copy(PymindfourModel *self)
{
    PymindfourModel *copy = (PymindfourModel *)PyObject_New(PymindfourModel, Py_TYPE(self));
    if (copy == NULL)
    {
        return NULL;
    }

    (void)mindfour_model_retain(self->handle);
    copy->handle = self->handle;
    return (PyObject *)copy;
}

static PyMethodDef PymindfourModel_methods[] = {
    {"__copy__", (PyCFunction)PymindfourModel_copy, METH_NOARGS, "Create a copy of the Model"},
    {NULL},
};

static PyObject *PymindfourModel_get_max_context_length(PymindfourModel *self, void *closure)
{
    size_t max_context_length = 0;
    const enum mindfour_status status = mindfour_model_get_max_context_length(self->handle, &max_context_length);
    if (status != mindfour_status_success)
    {

        return NULL;
    }

    return PyLong_FromSize_t(max_context_length);
}

static PyObject *PymindfourModel_get_tokenizer(PymindfourModel *self, void *closure)
{
    PyObject *args = PyTuple_Pack(1, self);
    if (args == NULL)
    {
        return NULL;
    }

    PyObject *tokenizer = PyObject_CallObject((PyObject *)&PymindfourTokenizer_Type, args);
    Py_DECREF(args);
    return tokenizer;
}

static PyGetSetDef PymindfourModel_getseters[] = {
    (PyGetSetDef){
        .name = "max_context_length",
        .get = (getter)PymindfourModel_get_max_context_length,
        .doc = "Maximum context length supported by the model",
    },
    (PyGetSetDef){
        .name = "tokenizer",
        .get = (getter)PymindfourModel_get_tokenizer,
        .doc = "Tokenizer object associated with the model",
    },
    {NULL}};

PyTypeObject PymindfourModel_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "mindfour.Model",
    .tp_basicsize = sizeof(PymindfourModel),
    .tp_flags = 0 | Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Model object",
    .tp_methods = PymindfourModel_methods,
    .tp_getset = PymindfourModel_getseters,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)PymindfourModel_init,
    .tp_dealloc = (destructor)PymindfourModel_dealloc,
};
