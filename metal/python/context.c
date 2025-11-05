#include <Python.h>

#include <mind-four.h>

#include "module.h"

static int PymindfourContext_init(PymindfourContext *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"model", "context_length", NULL};
    PyObject *model = NULL;
    Py_ssize_t context_length = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|$i", kwlist,
                                     &model, &context_length))
    {
        return -1;
    }
    if (!PyObject_TypeCheck(model, &PymindfourModel_Type))
    {
        PyErr_SetString(PyExc_TypeError, "model must be an mindfour.Model object");
        return -1;
    }
    if (context_length < 0)
    {
        PyErr_SetString(PyExc_ValueError, "context_length must be a positive integer");
        return -1;
    }

    enum mindfour_status status = mindfour_context_create(
        ((const PymindfourModel *)model)->handle,
        (size_t)context_length,
        &self->handle);
    if (status != mindfour_status_success)
    {
        goto error;
    }

    return 0;

error:
    mindfour_context_release(self->handle);
    self->handle = NULL;
    return -1;
}

static void PymindfourContext_dealloc(PymindfourContext *self)
{
    (void)mindfour_context_release(self->handle);
    self->handle = NULL;
    PyObject_Del((PyObject *)self);
}

static PyObject *PymindfourContext_copy(PymindfourContext *self)
{
    PymindfourContext *copy = (PymindfourContext *)PyObject_New(PymindfourContext, Py_TYPE(self));
    if (copy == NULL)
    {
        return NULL;
    }

    (void)mindfour_context_retain(self->handle);
    copy->handle = self->handle;
    return (PyObject *)copy;
}

static PyObject *PymindfourContext_append(PymindfourContext *self, PyObject *arg)
{
    if (PyBytes_Check(arg))
    {
        char *string_ptr = NULL;
        Py_ssize_t string_size = 0;
        if (PyBytes_AsStringAndSize(arg, &string_ptr, &string_size) < 0)
        {
            return NULL;
        }

        const enum mindfour_status status = mindfour_context_append_chars(
            self->handle, string_ptr, string_size, NULL);
        if (status != mindfour_status_success)
        {
            return NULL;
        }

        Py_RETURN_NONE;
    }
    else if (PyUnicode_Check(arg))
    {
        Py_ssize_t string_size = 0;
        const char *string_ptr = PyUnicode_AsUTF8AndSize(arg, &string_size);
        if (string_ptr == NULL)
        {
            return NULL;
        }

        const enum mindfour_status status = mindfour_context_append_chars(
            self->handle, string_ptr, string_size, NULL);
        if (status != mindfour_status_success)
        {

            return NULL;
        }

        Py_RETURN_NONE;
    }
    else if (PyLong_Check(arg))
    {
        const unsigned long token_as_ulong = PyLong_AsUnsignedLong(arg);
        if (token_as_ulong == (unsigned long)-1 && PyErr_Occurred())
        {
            return NULL;
        }

        const uint32_t token = (uint32_t)token_as_ulong;
        const enum mindfour_status status = mindfour_context_append_tokens(
            self->handle, 1, &token);
        if (status != mindfour_status_success)
        {
            return NULL;
        }

        Py_RETURN_NONE;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "expected a bytes or integer argument");
        return NULL;
    }
}

static PyObject *PymindfourContext_process(PymindfourContext *self)
{
    const enum mindfour_status status = mindfour_context_process(self->handle);
    if (status != mindfour_status_success)
    {

        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *PymindfourContext_sample(PymindfourContext *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"temperature", "seed", NULL};

    unsigned long long seed = 0;
    float temperature = 1.0f;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|$fK", kwlist,
                                     &temperature, &seed))
    {
        return NULL;
    }

    uint32_t token_out = UINT32_MAX;
    enum mindfour_status status = mindfour_context_sample(
        self->handle, temperature, (uint64_t)seed, &token_out);
    if (status != mindfour_status_success)
    {

        return NULL;
    }

    return PyLong_FromUnsignedLong((unsigned long)token_out);
}

static PyObject *PymindfourContext_reset(PymindfourContext *self)
{
    const enum mindfour_status status = mindfour_context_reset(self->handle);
    if (status != mindfour_status_success)
    {

        return NULL;
    }

    Py_RETURN_NONE;
}

static PyMethodDef PymindfourContext_methods[] = {
    {"__copy__", (PyCFunction)PymindfourContext_copy, METH_NOARGS, "Create a copy of the Context"},
    {"append", (PyCFunction)PymindfourContext_append, METH_O, "Append bytes to the Context"},
    {"process", (PyCFunction)PymindfourContext_process, METH_NOARGS, "Process tokens in the Context"},
    {"sample", (PyCFunction)PymindfourContext_sample, METH_VARARGS | METH_KEYWORDS, "Sample token prediction from the Context"},
    {"reset", (PyCFunction)PymindfourContext_reset, METH_NOARGS, "Discard the content of the Context"},
    {NULL},
};

static PyObject *PymindfourContext_get_num_tokens(PymindfourContext *self, void *closure)
{
    size_t num_tokens = 0;
    const enum mindfour_status status = mindfour_context_get_num_tokens(self->handle, &num_tokens);
    if (status != mindfour_status_success)
    {

        return NULL;
    }

    return PyLong_FromSize_t(num_tokens);
}

static PyObject *PymindfourContext_get_max_tokens(PymindfourContext *self, void *closure)
{
    size_t max_tokens = 0;
    const enum mindfour_status status = mindfour_context_get_max_tokens(self->handle, &max_tokens);
    if (status != mindfour_status_success)
    {

        return NULL;
    }

    return PyLong_FromSize_t(max_tokens);
}

static PyObject *PymindfourContext_get_tokens(PymindfourContext *self, void *closure)
{
    PyObject *token_list_obj = NULL;
    PyObject *token_obj = NULL;
    uint32_t *token_ptr = NULL;

    size_t num_tokens = 0;
    mindfour_context_get_tokens(self->handle, NULL, 0, &num_tokens);

    if (num_tokens != 0)
    {
        token_ptr = (uint32_t *)PyMem_Malloc(num_tokens * sizeof(uint32_t));
        if (token_ptr == NULL)
        {

            goto error;
        }

        enum mindfour_status status = mindfour_context_get_tokens(self->handle, token_ptr, num_tokens, &num_tokens);
        if (status != mindfour_status_success)
        {

            goto error;
        }
    }

    token_list_obj = PyList_New((Py_ssize_t)num_tokens);
    if (token_list_obj == NULL)
    {
        goto error;
    }

    for (size_t t = 0; t < num_tokens; t++)
    {
        token_obj = PyLong_FromUnsignedLong((unsigned long)token_ptr[t]);
        if (token_obj == NULL)
        {
            goto error;
        }
        if (PyList_SetItem(token_list_obj, (Py_ssize_t)t, token_obj) < 0)
        {
            goto error;
        }
        token_obj = NULL;
    }

    PyMem_Free(token_ptr);
    return token_list_obj;

error:
    PyMem_Free(token_ptr);
    Py_XDECREF(token_obj);
    Py_XDECREF(token_list_obj);
    return NULL;
}

static PyGetSetDef PymindfourContext_getseters[] = {
    (PyGetSetDef){
        .name = "num_tokens",
        .get = (getter)PymindfourContext_get_num_tokens,
        .doc = "Current number of tokens in the context",
    },
    (PyGetSetDef){
        .name = "max_tokens",
        .get = (getter)PymindfourContext_get_max_tokens,
        .doc = "Maximum number of tokens in the context",
    },
    (PyGetSetDef){
        .name = "tokens",
        .get = (getter)PymindfourContext_get_tokens,
        .doc = "List of token IDs in the context",
    },
    {NULL}};

PyTypeObject PymindfourContext_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "mindfour.Context",
    .tp_basicsize = sizeof(PymindfourContext),
    .tp_flags = 0 | Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Context object",
    .tp_methods = PymindfourContext_methods,
    .tp_getset = PymindfourContext_getseters,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)PymindfourContext_init,
    .tp_dealloc = (destructor)PymindfourContext_dealloc,
};
