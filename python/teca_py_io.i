%{
#include "teca_algorithm.h"
#include "teca_cf_reader.h"
#include "teca_multi_cf_reader.h"
#include "teca_wrf_reader.h"
#include "teca_array_collection_reader.h"
#include "teca_cf_writer.h"
#include "teca_file_util.h"
#include "teca_table_reader.h"
#include "teca_table_writer.h"
#include "teca_cartesian_mesh_reader.h"
#include "teca_cartesian_mesh_writer.h"
%}

/***************************************************************************
 cf_reader
 ***************************************************************************/
#ifdef TECA_HAS_NETCDF
%ignore teca_cf_reader::shared_from_this;
%shared_ptr(teca_cf_reader)
%ignore teca_cf_reader::operator=;
%include "teca_cf_reader.h"
#endif

/***************************************************************************
 multi_cf_reader
 ***************************************************************************/
#ifdef TECA_HAS_NETCDF
%ignore teca_multi_cf_reader::shared_from_this;
%shared_ptr(teca_multi_cf_reader)
%ignore teca_multi_cf_reader::operator=;
%ignore teca_multi_cf_reader::get_variables(std::vector<std::string>&);
%include "teca_multi_cf_reader.h"
#endif
%extend teca_multi_cf_reader
{
    PyObject *get_variables()
    {
        teca_py_gil_state gil;

        std::vector<std::string> vars;
        $self->get_variables(vars);

        size_t n_vars = vars.size();

        PyObject *vars_out = PyList_New(n_vars);

        for (size_t i = 0; i < n_vars; ++i)
            PyList_SetItem(vars_out, i, CStringToPyString(vars[i].c_str()));

        return vars_out;
    }
}

/***************************************************************************
 wrf_reader
 ***************************************************************************/
#ifdef TECA_HAS_NETCDF
%ignore teca_wrf_reader::shared_from_this;
%shared_ptr(teca_wrf_reader)
%ignore teca_wrf_reader::operator=;
%include "teca_wrf_reader.h"
#endif

/***************************************************************************
 array_collection_reader
 ***************************************************************************/
#ifdef TECA_HAS_NETCDF
%ignore teca_array_collection_reader::shared_from_this;
%shared_ptr(teca_array_collection_reader)
%ignore teca_array_collection_reader::operator=;
%include "teca_array_collection_reader.h"
#endif

/***************************************************************************
 cf_writer
 ***************************************************************************/
#ifdef TECA_HAS_NETCDF
%ignore teca_cf_writer::shared_from_this;
%shared_ptr(teca_cf_writer)
%ignore teca_cf_writer::operator=;
%include "teca_cf_writer.h"
#endif

/***************************************************************************
 table_reader
 ***************************************************************************/
%ignore teca_table_reader::shared_from_this;
%shared_ptr(teca_table_reader)
%ignore teca_table_reader::operator=;
%include "teca_table_reader.h"

/***************************************************************************
 table_writer
 ***************************************************************************/
%ignore teca_table_writer::shared_from_this;
%shared_ptr(teca_table_writer)
%ignore teca_table_writer::operator=;
%include "teca_table_writer.h"

/***************************************************************************
 cartesian_mesh_reader
 ***************************************************************************/
%ignore teca_cartesian_mesh_reader::shared_from_this;
%shared_ptr(teca_cartesian_mesh_reader)
%ignore teca_cartesian_mesh_reader::operator=;
%include "teca_cartesian_mesh_reader.h"

/***************************************************************************
 cartesian_mesh_writer
 ***************************************************************************/
%ignore teca_cartesian_mesh_writer::shared_from_this;
%shared_ptr(teca_cartesian_mesh_writer)
%ignore teca_cartesian_mesh_writer::operator=;
%include "teca_cartesian_mesh_writer.h"


/***************************************************************************
 utility functions
 ***************************************************************************/
%inline
%{
struct file_util
{
static
std::string replace_timestep(const std::string &file_name, unsigned long time_step, int width = 6)
{
    std::string tmp = file_name;
    teca_file_util::replace_timestep(tmp, time_step, width);
    return tmp;
}

static
std::string replace_time(const std::string &file_name, double t,
    const std::string &calendar, const std::string &units,
    const std::string &format)
{
    std::string tmp = file_name;
    if (teca_file_util::replace_time(tmp, t, calendar, units, format))
    {
        TECA_PY_ERROR(PyExc_RuntimeError,
            "Failed to replace time in \"" << file_name << "\" with t "
            << t << "calendar \"" << calendar << "\" units \"" << units
            << "\" and format \"" << format << "\"")
        return "";
    }
    return tmp;
}

static
std::string replace_extension(const std::string &file_name, const std::string &ext)
{
    std::string tmp = file_name;
    teca_file_util::replace_extension(tmp, ext);
    return tmp;
}
};
%}
