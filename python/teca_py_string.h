#ifndef teca_py_string_h
#define teca_py_string_h

#include "teca_config.h"

#if TECA_PYTHON_VERSION == 2

#define PyStringCheck PyString_Check
#define PyStringToCString PyString_AsString
#define CStringToPyString PyString_FromString


#elif TECA_PYTHON_VERSION == 3

#define PyStringCheck PyUnicode_Check
#define PyStringToCString PyUnicode_AsUTF8
#define CStringToPyString PyUnicode_FromString

#else
#error #TECA_PYTHON_VERSION " must be 2 or 3"
#endif

#endif
