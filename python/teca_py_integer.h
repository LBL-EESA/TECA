#ifndef teca_py_integer_h
#define teca_py_integer_h

#include "teca_config.h"

#if TECA_PYTHON_VERSION == 2

#define PyIntegerCheck PyInt_Check
#define PyLongCheck PyLong_Check

#define PyLongToCLong PyLong_AsLong
#define CIntToPyInteger PyInt_FromLong
#define CIntUToPyInteger PyInt_FromSize_t
#define CIntLLToPyInteger PyLong_FromLongLong
#define CIntULLToPyInteger PyLong_FromUnsignedLongLong

#define PyIntegerToCInt PyInt_AsLong
#define PyIntegerToCIntU PyInt_AsSize_t
#define PyIntegerToCIntLL PyInt_AsLongLong
#define PyIntegerToCIntULL PyInt_AsUnsignedLongLong

#elif TECA_PYTHON_VERSION == 3

#define PyIntegerCheck PyLong_Check
#define PyLongCheck PyLong_Check

#define PyLongToCLong PyLong_AsLong
#define PyIntegerToCInt PyLong_AsLong
#define PyIntegerToCIntU PyLong_AsUnsignedLong
#define PyIntegerToCIntLL PyLong_AsLongLong
#define PyIntegerToCIntULL PyLong_AsUnsignedLongLong

#define CIntToPyInteger PyLong_FromLong
#define CIntUToPyInteger PyLong_FromUnsignedLong
#define CIntLLToPyInteger PyLong_FromLongLong
#define CIntULLToPyInteger PyLong_FromUnsignedLongLong

#else
#error #TECA_PYTHON_VERSION " must be 2 or 3"
#endif

#endif
