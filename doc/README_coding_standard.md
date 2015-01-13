
tabs, spaces, and indentation
=============================
* use no \t chars
* leave no trailing white space
* indent is 4 spaces
* use a single space between operators
    eg:
        a < b

braces
======
* use standrad C++ bracing, braces go on the previous
  indentation level
  eg:
    if (a < b)
    {
        foo(a,b);
    }

variable, funciton, and class names
===================================
* use lower case with _ separator
  eg:
        my_var_name

macros
======
* to be definied in all caps, and do not require a semicolon
  eg:
        DO_SOMETHING(blah, blah)

warnings
========
    treat warnings as errors, developers should use -Wall or
    the compiler equivalent ( -Weverything, /W4, etc)

conditionals
============
    conditionals may omit braces, in that case the code
    should be on the following line (ie no onliners)
    eg:
        if (a < b)
            foo(a,b);
headers
=======
* don't include using directives
* use include guards of the format
    #ifndef file_name_h
    #define file_name_h
    ...
    #endif

classes
=======
* use this pointer to access member variables, unless it's a
  private member variable prepended with m_
* use this pointer to call member functions
* use pimpl idiom for base classes and
  use forward declarations to limit includes
* use std::shared_ptr in place of C style pointers for heavy objects
* use teca's convenience macros where possible
* use const qualifiers when ever possible


reporting errors
================
* from functions return a non-zero value
* use the error macro
    eg:
        TECA_ERROR("message " << var)

exceptions and exception safety
===============================
* the code is NOT designed for exception safety.
* use execptions only when absolutely necessary and the program
  needs to terminate.

threading
=========
* ?design for thread per-pipeline invocation?
* ?algorithms need to be thread safe?
