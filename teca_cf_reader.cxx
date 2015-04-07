#include "teca_cf_reader.h"
#include "teca_file_util.h"

#include <netcdf.h>
#include <iostream>
#include <algorithm>

using std::vector;
using std::string;
using std::endl;
using std::cerr;

using teca_file_util::strip_path_from_filename;
using teca_file_util::strip_filename_from_path;
using teca_file_util::locate_files;

// macro to help with netcdf data types
#define NC_DISPATCH_FP(tc_, code_)                          \
    switch (tc_)                                            \
    {                                                       \
    NC_DISPATCH_CASE(NC_FLOAT, float, code_)                \
    NC_DISPATCH_CASE(NC_DOUBLE, double, code_)              \
    default:                                                \
        TECA_ERROR("netcdf type code_ " << tc_              \
            << " is not a floating point type")             \
    }
#define NC_DISPATCH(tc_, code_)                             \
    switch (tc_)                                            \
    {                                                       \
    NC_DISPATCH_CASE(NC_BYTE, char, code_)                  \
    NC_DISPATCH_CASE(NC_UBYTE, unsigned char, code_)        \
    NC_DISPATCH_CASE(NC_CHAR, char, code_)                  \
    NC_DISPATCH_CASE(NC_SHORT, short int, code_)            \
    NC_DISPATCH_CASE(NC_USHORT, unsigned short int, code_)  \
    NC_DISPATCH_CASE(NC_INT, int, code_)                    \
    NC_DISPATCH_CASE(NC_UINT, unsigned int, code_)          \
    NC_DISPATCH_CASE(NC_INT64, long long, code_)            \
    NC_DISPATCH_CASE(NC_UINT64, unsigned long long, code_)  \
    NC_DISPATCH_CASE(NC_FLOAT, float, code_)                \
    NC_DISPATCH_CASE(NC_DOUBLE, double, code_)              \
    NC_DISPATCH_CASE(NC_STRING, char **, code_)             \
    default:                                                \
        TECA_ERROR("netcdf type code_ " << tc_              \
            << " is not supported")                         \
    }
#define NC_DISPATCH_CASE(cc_, tt_, code_)   \
    case cc_:                               \
    {                                       \
        using NC_T = tt_;                   \
        code_                               \
        break;                              \
    }

// --------------------------------------------------------------------------
teca_cf_reader::teca_cf_reader() :
    files_regex(""),
    x_axis_variable("lon"),
    y_axis_variable("lat"),
    z_axis_variable(""),
    t_axis_variable("time")
{}

// --------------------------------------------------------------------------
teca_metadata teca_cf_reader::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
    cerr << "teca_cf_reader::get_output_metadata" << endl;

    teca_metadata output_md;

    vector<string> files;

    string regex = strip_path_from_filename(this->files_regex);
    string path = strip_filename_from_path(this->files_regex);

    if (teca_file_util::locate_files(path, regex, files))
    {
        TECA_ERROR(
            << "Failed to locate any files." << endl
            << this->files_regex << endl
            << path << endl
            << regex)
        return teca_metadata();
    }

    int ierr = 0;
    int file_id = 0;
    string file = path + PATH_SEP + files[0];

    // get mesh coordinates and dimensions
    int x_id = 0;
    int y_id = 0;
    int z_id = 0;
    int t_id = 0;
    size_t n_x = 0;
    size_t n_y = 0;
    size_t n_z = 0;
    size_t n_t = 0;
    nc_type x_t = 0;
    nc_type y_t = 0;
    nc_type z_t = 0;
    nc_type t_t = 0;
    int n_vars = 0;

    if (((ierr = nc_open(file.c_str(), NC_NOWRITE, &file_id)) != NC_NOERR)
        || ((ierr = nc_inq_dimid(file_id, x_axis_variable.c_str(), &x_id)) != NC_NOERR)
        || ((ierr = nc_inq_dimlen(file_id, x_id, &n_x)) != NC_NOERR)
        || ((ierr = nc_inq_varid(file_id, x_axis_variable.c_str(), &x_id)) != NC_NOERR)
        || ((ierr = nc_inq_vartype(file_id, x_id, &x_t)) != NC_NOERR)
        || ((ierr = nc_inq_dimid(file_id, y_axis_variable.c_str(), &y_id)) != NC_NOERR)
        || ((ierr = nc_inq_dimlen(file_id, y_id, &n_y)) != NC_NOERR)
        || ((ierr = nc_inq_varid(file_id, y_axis_variable.c_str(), &y_id)) != NC_NOERR)
        || ((ierr = nc_inq_vartype(file_id, y_id, &y_t)) != NC_NOERR)
        || (!z_axis_variable.empty()
        && (((ierr = nc_inq_dimid(file_id, z_axis_variable.c_str(), &z_id)) != NC_NOERR)
        || ((ierr = nc_inq_dimlen(file_id, z_id, &n_x)) != NC_NOERR)
        || ((ierr = nc_inq_varid(file_id, z_axis_variable.c_str(), &z_id)) != NC_NOERR)
        || ((ierr = nc_inq_vartype(file_id, z_id, &z_t)) != NC_NOERR)))
        || ((ierr = nc_inq_dimid(file_id, t_axis_variable.c_str(), &t_id)) != NC_NOERR)
        || ((ierr = nc_inq_dimlen(file_id, t_id, &n_t)) != NC_NOERR)
        || ((ierr = nc_inq_varid(file_id, t_axis_variable.c_str(), &t_id)) != NC_NOERR)
        || ((ierr = nc_inq_vartype(file_id, t_id, &t_t)) != NC_NOERR)
        || ((ierr = nc_inq_nvars(file_id, &n_vars)) != NC_NOERR))
    {
        nc_close(file_id);
        TECA_ERROR(
            << "Failed to query mesh properties, " << file << endl
            << nc_strerror(ierr))
        return teca_metadata();
    }

    // enumerate mesh arrays and their attributes
    vector<string> vars;
    for (int i = 0; i < n_vars; ++i)
    {
        char var_name[NC_MAX_NAME + 1] = {'\0'};
        nc_type var_type = 0;
        int n_dims = 0;
        int dim_id[NC_MAX_VAR_DIMS] = {0};
        int n_atts = 0;

        if ((ierr = nc_inq_var(file_id, i, var_name, &var_type, &n_dims, dim_id, &n_atts)) != NC_NOERR)
        {
            nc_close(file_id);
            TECA_ERROR(
                << "Failed to query " << i << "th variable, "
                << file << endl << nc_strerror(ierr))
            return teca_metadata();
        }

        // skip scalars
        if (n_dims == 0)
            continue;

        vector<size_t> dims;
        vector<string> dim_names;
        for (int ii = 0; ii < n_dims; ++ii)
        {
            char dim_name[NC_MAX_NAME + 1] = {'\0'};
            size_t dim = 0;
            if ((ierr = nc_inq_dim(file_id, dim_id[ii], dim_name, &dim)) != NC_NOERR)
            {
                nc_close(file_id);
                TECA_ERROR(
                    << "Failed to query " << ii << "th dimension of variable, "
                    << var_name << ", " << file << endl << nc_strerror(ierr))
                return teca_metadata();
            }

            // skip dimensions
            //if (string(var_name) == dim_name)
            //    break;

            dim_names.push_back(dim_name);
            dims.push_back(dim);
        }

        if (dims.size() == 0)
            continue;

        vars.push_back(var_name);

        teca_metadata atts;
        atts.set("dims", dims);
        atts.set("dim_names", dim_names);

        char *buffer = nullptr;
        for (int ii = 0; ii < n_atts; ++ii)
        {
            char att_name[NC_MAX_NAME + 1] = {'\0'};
            nc_type att_type = 0;
            size_t att_len = 0;
            if (((ierr = nc_inq_attname(file_id, i, ii, att_name)) != NC_NOERR)
                || ((ierr = nc_inq_att(file_id, i, att_name, &att_type, &att_len)) != NC_NOERR))
            {
                nc_close(file_id);
                TECA_ERROR(
                    << "Failed to query " << ii << "th attribute of variable, "
                    << var_name << ", " << file << endl << nc_strerror(ierr))
                return teca_metadata();
            }
            if (att_type == NC_CHAR)
            {
                buffer = static_cast<char*>(realloc(buffer, att_len + 1));
                buffer[att_len] = '\0';
                nc_get_att_text(file_id, i, att_name, buffer);
                atts.set(att_name, string(buffer));
            }
        }
        free(buffer);

        output_md.set(var_name, atts);
    }

    output_md.set("variables", vars);

    // read coordinate arrays
    p_teca_variant_array x_axis;
    NC_DISPATCH_FP(x_t,
        size_t x_0 = 0;
        p_teca_variant_array_impl<NC_T> x = teca_variant_array_impl<NC_T>::New(n_x);
        if ((ierr = nc_get_vara(file_id, x_id, &x_0, &n_x, x->get())) != NC_NOERR)
        {
            nc_close(file_id);
            TECA_ERROR(
                << "Failed to read x axis, " << x_axis_variable << endl
                << file << endl << nc_strerror(ierr))
            return teca_metadata();
        }
        x_axis = x;
        )

    p_teca_variant_array y_axis;
    NC_DISPATCH_FP(y_t,
        size_t y_0 = 0;
        p_teca_variant_array_impl<NC_T> y = teca_variant_array_impl<NC_T>::New(n_y);
        if ((ierr = nc_get_vara(file_id, y_id, &y_0, &n_y, y->get())) != NC_NOERR)
        {
            nc_close(file_id);
            TECA_ERROR(
                << "Failed to read y axis, " << y_axis_variable << endl
                << file << endl << nc_strerror(ierr))
            return teca_metadata();
        }
        y_axis = y;
        )

    p_teca_variant_array z_axis;
    if (!z_axis_variable.empty())
    {
        NC_DISPATCH_FP(z_t,
            size_t z_0 = 0;
            p_teca_variant_array_impl<NC_T> z = teca_variant_array_impl<NC_T>::New(n_z);
            if ((ierr = nc_get_vara(file_id, z_id, &z_0, &n_z, z->get())) != NC_NOERR)
            {
                nc_close(file_id);
                TECA_ERROR(
                    << "Failed to read z axis, " << z_axis_variable << endl
                    << file << endl << nc_strerror(ierr))
                return teca_metadata();
            }
            z_axis = z;
            )
    }

    // collect time steps from this and the rest of the files
    p_teca_variant_array t_axis;
    if (!t_axis_variable.empty())
    {
        NC_DISPATCH_FP(t_t,
            size_t t_0 = 0;
            p_teca_variant_array_impl<NC_T> t = teca_variant_array_impl<NC_T>::New(n_t);
            if ((ierr = nc_get_vara(file_id, t_id, &t_0, &n_t, t->get())) != NC_NOERR)
            {
                nc_close(file_id);
                TECA_ERROR(
                    << "Failed to read t axis, " << t_axis_variable << endl
                    << file << endl << nc_strerror(ierr))
                return teca_metadata();
            }
            t_axis = t;
            )
    }

    size_t n_files = files.size();
    for (size_t i = 1; i < n_files; ++i)
    {
        string file = path + PATH_SEP + files[i];
        if (((ierr = nc_open(file.c_str(), NC_NOWRITE, &file_id)) != NC_NOERR)
            || ((ierr = nc_inq_dimid(file_id, t_axis_variable.c_str(), &t_id)) != NC_NOERR)
            || ((ierr = nc_inq_dimlen(file_id, t_id, &n_t)) != NC_NOERR)
            || ((ierr = nc_inq_varid(file_id, t_axis_variable.c_str(), &t_id)) != NC_NOERR)
            || ((ierr = nc_inq_vartype(file_id, t_id, &t_t)) != NC_NOERR))
        {
            nc_close(file_id);
            TECA_ERROR(
                << "Failed to query time in the " << i << "th file, "
                << file << endl << nc_strerror(ierr))
            return teca_metadata();
        }
        NC_DISPATCH_FP(t_t,
            size_t t_0 = 0;
            p_teca_variant_array_impl<NC_T> t
                = std::dynamic_pointer_cast<teca_variant_array_impl<NC_T>>(t_axis);
            if (!t)
            {
                nc_close(file_id);
                TECA_ERROR(
                    << t_axis_variable << " has in compatible type in the "
                    << i << "th file, " << file)
                return teca_metadata();
            }
            size_t n = t->size();
            t->resize(n + n_t);
            if ((ierr = nc_get_vara(file_id, t_id, &t_0, &n_t, t->get()+n)) != NC_NOERR)
            {
                nc_close(file_id);
                TECA_ERROR(
                    << "Failed to read t axis, " << t_axis_variable << endl
                    << file << endl << nc_strerror(ierr))
                return teca_metadata();
            }
            )
        nc_close(file_id);
    }

    teca_metadata coords;
    coords.set("x_variable", x_axis_variable);
    coords.set("y_variable", y_axis_variable);
    coords.set("z_variable", z_axis_variable);
    coords.set("t_variable", t_axis_variable);
    coords.set("x", x_axis);
    coords.set("y", y_axis);
    coords.set("z", z_axis);
    coords.set("t", t_axis);

    vector<size_t> whole_extent(6, 0);
    whole_extent[1] = n_x - 1;
    whole_extent[3] = n_y - 1;
    whole_extent[5] = n_z - 1;
    coords.set("whole_extent", whole_extent);

    output_md.set("coordinates", coords);

    return output_md;
}

// --------------------------------------------------------------------------
p_teca_dataset teca_cf_reader::execute(
    unsigned int port,
    const std::vector<p_teca_dataset> &input_data,
    const teca_metadata &request)
{

    return nullptr;
}
