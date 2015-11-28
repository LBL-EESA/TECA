%define MDOC
"TECA core module

TODO -- documentation
"
%enddef

%module (docstring=MDOC) py_teca_core

%{
#include <sstream>
#include <vector>
#include <Python.h>
#include "teca_metadata.h"
#include "teca_variant_array.h"
%}

%include "py_teca_buffer.i"
%include "std_string.i"

/*
 * teca_metadata
 */
%define teca_metadata_api(c_type, c_type_name)
void declare_ ## c_type_name(const std::string &name)
{ self->declare<c_type>(name); }

void insert_ ## c_type_name(const std::string &name, const std::vector<c_type> &md)
{ self->insert(name, md); }

void append_ ## c_type_name(const std::string &name, const std::vector<c_type> &md)
{ self->append(name, md); }

void set_ ## c_type_name(const std::string &name, c_type idx, c_type val)
{ self->set(name, idx, val); }

c_type get_ ## c_type_name(const std::string &name, c_type idx)
{
    c_type tmp = 0;
    if (self->get(name, idx, tmp))
        TECA_ERROR("Failed to get " << idx << "ith value of " << name)
    return tmp;
}
%enddef
%ignore teca_metadata::teca_metadata(teca_metadata &&);
%ignore teca_metadata::operator=;
%ignore operator<(const teca_metadata &, const teca_metadata &);
%ignore operator&(const teca_metadata &, const teca_metadata &);
%ignore operator==(const teca_metadata &, const teca_metadata &);
%ignore operator!=(const teca_metadata &, const teca_metadata &);
%include "teca_metadata.h"
%extend teca_metadata
{
    const char *__str__()
    {
        static std::ostringstream oss;
        oss.str("");
        self->to_stream(oss);
        return oss.str().c_str();
    }
    teca_metadata_api(int,i)
    teca_metadata_api(long long,ll)
    teca_metadata_api(float,f)
    teca_metadata_api(double,d)
}

/*
 * teca_variant_array
 *
%include <std_shared_ptr.i>
%shared_ptr(teca_variant_array)
%shared_ptr(teca_variant_array_impl<char>)
%shared_ptr(teca_variant_array_impl<int>)
%shared_ptr(teca_variant_array_impl<long long>)
%shared_ptr(teca_variant_array_impl<float>)
%shared_ptr(teca_variant_array_impl<double>)
%inline{
typedef std::shared_ptr<teca_variant_array> p_teca_variant_array;
}
%include "teca_common.h"
%include "teca_shared_object.h"
%include "teca_variant_array_fwd.h"
%ignore teca_variant_array::operator=;
%include "teca_variant_array.h"
%template(teca_variant_array_char) teca_variant_array_impl<char>;
%template(teca_variant_array_int) teca_variant_array_impl<int>;
%template(teca_variant_array_int64) teca_variant_array_impl<long long>;
%template(teca_variant_array_float) teca_variant_array_impl<float>;
%template(teca_variant_array_double) teca_variant_array_impl<double>;
%extend teca_variant_array
{
    const char *__str__()
    {
        static std::ostringstream oss;
        oss.str("");
        self->to_stream(oss);
        return oss.str().c_str();
    }
}*/
