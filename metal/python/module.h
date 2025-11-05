#include <Python.h>

#include <mind-four.h>

typedef struct
{
    PyObject_HEAD
        mindfour_model_t handle;
} PymindfourModel;

typedef struct
{
    PyObject_HEAD
        mindfour_tokenizer_t handle;
} PymindfourTokenizer;

typedef struct
{
    PyObject_HEAD
        mindfour_context_t handle;
} PymindfourContext;

extern PyTypeObject PymindfourModel_Type;
extern PyTypeObject PymindfourTokenizer_Type;
extern PyTypeObject PymindfourContext_Type;
