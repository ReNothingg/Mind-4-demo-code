#include <Python.h>

#include <mind-four.h>

#include "module.h"

static PyObject *PymindfourTokenizer_new(PyTypeObject *subtype, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"model", NULL};
    PyObject *model = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PymindfourModel_Type, &model))
    {
        return NULL;
    }

    PymindfourTokenizer *self = (PymindfourTokenizer *)subtype->tp_alloc(subtype, 0);
    if (self == NULL)
    {
        return NULL;
    }

    const enum mindfour_status status = mindfour_model_get_tokenizer(
        ((const PymindfourModel *)model)->handle,
        &self->handle);
    if (status != mindfour_status_success)
    {

        return NULL;
    }

    return (PyObject *)self;
}

static void PymindfourTokenizer_dealloc(PymindfourTokenizer *self)
{
    (void)mindfour_tokenizer_release(self->handle);
    self->handle = NULL;
    PyObject_Del((PyObject *)self);
}

static PyObject *PymindfourTokenizer_copy(PymindfourTokenizer *self)
{
    PymindfourTokenizer *copy = (PymindfourTokenizer *)PyObject_New(PymindfourTokenizer, Py_TYPE(self));
    if (copy == NULL)
    {
        return NULL;
    }

    (void)mindfour_tokenizer_retain(self->handle);
    copy->handle = self->handle;
    return (PyObject *)copy;
}

static PyObject *PymindfourTokenizer_encode_special_token(PymindfourTokenizer *self, PyObject *arg)
{
    if (PyUnicode_Check(arg))
    {
        const char *string_ptr = PyUnicode_AsUTF8(arg);
        if (string_ptr == NULL)
        {
            return NULL;
        }

        enum mindfour_special_token token_type = mindfour_special_token_invalid;
        if (strcmp(string_ptr, "<|return|>") == 0)
        {
            token_type = mindfour_special_token_return;
        }
        else if (strcmp(string_ptr, "<|start|>") == 0)
        {
            token_type = mindfour_special_token_start;
        }
        else if (strcmp(string_ptr, "<|message|>") == 0)
        {
            token_type = mindfour_special_token_message;
        }
        else if (strcmp(string_ptr, "<|end|>") == 0)
        {
            token_type = mindfour_special_token_end;
        }
        else if (strcmp(string_ptr, "<|refusal|>") == 0)
        {
            token_type = mindfour_special_token_refusal;
        }
        else if (strcmp(string_ptr, "<|constrain|>") == 0)
        {
            token_type = mindfour_special_token_constrain;
        }
        else if (strcmp(string_ptr, "<|channel|>") == 0)
        {
            token_type = mindfour_special_token_channel;
        }
        else if (strcmp(string_ptr, "<|call|>") == 0)
        {
            token_type = mindfour_special_token_call;
        }
        else if (strcmp(string_ptr, "<|untrusted|>") == 0)
        {
            token_type = mindfour_special_token_untrusted;
        }
        else if (strcmp(string_ptr, "<|end_untrusted|>") == 0)
        {
            token_type = mindfour_special_token_end_untrusted;
        }
        else
        {
            PyErr_Format(PyExc_ValueError, "unrecognized special token: %s", string_ptr);
            return NULL;
        }

        uint32_t token_id = UINT32_MAX;
        const enum mindfour_status status = mindfour_tokenizer_get_special_token_id(
            self->handle, token_type, &token_id);
        if (status != mindfour_status_success || token_id == UINT32_MAX)
        {
            PyErr_Format(PyExc_ValueError, "tokenizer does not support the %s token", string_ptr);
            return NULL;
        }

        return PyLong_FromUnsignedLong((unsigned long)token_id);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "string argument expected");
        return NULL;
    }
}

static PyObject *PymindfourTokenizer_decode(PymindfourTokenizer *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"token", NULL};
    unsigned int token = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "I", kwlist, &token))
    {
        return NULL;
    }

    const void *token_ptr = NULL;
    size_t token_size = 0;
    const enum mindfour_status status = mindfour_tokenizer_decode(self->handle, (uint32_t)token, &token_ptr, &token_size);
    if (status != mindfour_status_success)
    {

        return NULL;
    }

    return PyBytes_FromStringAndSize((const char *)token_ptr, (Py_ssize_t)token_size);
}

static PyMethodDef PymindfourTokenizer_methods[] = {
    {"__copy__", (PyCFunction)PymindfourTokenizer_copy, METH_NOARGS, "Create a copy of the Tokenizer"},
    {"encode_special_token", (PyCFunction)PymindfourTokenizer_encode_special_token, METH_O, "Query ID of a special token"},
    {"decode", (PyCFunction)PymindfourTokenizer_decode, METH_VARARGS | METH_KEYWORDS, "Convert text token ID to bytes"},
    {NULL},
};

static PyObject *PymindfourTokenizer_get_num_text_tokens(PymindfourTokenizer *self, void *closure)
{
    uint32_t num_text_tokens = 0;
    const enum mindfour_status status = mindfour_tokenizer_get_num_text_tokens(self->handle, &num_text_tokens);
    if (status != mindfour_status_success)
    {

        return NULL;
    }

    return PyLong_FromUnsignedLong((unsigned long)num_text_tokens);
}

static PyObject *PymindfourTokenizer_get_num_special_tokens(PymindfourTokenizer *self, void *closure)
{
    uint32_t num_special_tokens = 0;
    const enum mindfour_status status = mindfour_tokenizer_get_num_special_tokens(self->handle, &num_special_tokens);
    if (status != mindfour_status_success)
    {

        return NULL;
    }

    return PyLong_FromUnsignedLong((unsigned long)num_special_tokens);
}

static PyObject *PymindfourTokenizer_get_num_tokens(PymindfourTokenizer *self, void *closure)
{
    uint32_t num_tokens = 0;
    const enum mindfour_status status = mindfour_tokenizer_get_num_tokens(self->handle, &num_tokens);
    if (status != mindfour_status_success)
    {

        return NULL;
    }

    return PyLong_FromUnsignedLong((unsigned long)num_tokens);
}

static PyGetSetDef PymindfourTokenizer_getseters[] = {
    (PyGetSetDef){
        .name = "num_tokens",
        .get = (getter)PymindfourTokenizer_get_num_tokens,
        .doc = "Total number of tokens in the tokenizer dictionary",
    },
    (PyGetSetDef){
        .name = "num_text_tokens",
        .get = (getter)PymindfourTokenizer_get_num_text_tokens,
        .doc = "Number of text tokens in the tokenizer dictionary",
    },
    (PyGetSetDef){
        .name = "num_special_tokens",
        .get = (getter)PymindfourTokenizer_get_num_special_tokens,
        .doc = "Number of special tokens in the tokenizer dictionary",
    },
    {NULL}};

PyTypeObject PymindfourTokenizer_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "mindfour.Tokenizer",
    .tp_basicsize = sizeof(PymindfourTokenizer),
    .tp_flags = 0 | Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Tokenizer object",
    .tp_methods = PymindfourTokenizer_methods,
    .tp_getset = PymindfourTokenizer_getseters,
    .tp_new = PymindfourTokenizer_new,
    .tp_dealloc = (destructor)PymindfourTokenizer_dealloc,
};
