#include "teca_table_writer.h"

#include "teca_table.h"
#include "teca_database.h"
#include "teca_metadata.h"
#include "teca_binary_stream.h"
#include "teca_file_util.h"
#include "teca_mpi.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <cerrno>
#include <cstring>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#if defined(TECA_HAS_LIBXLSXWRITER)
#include <xlsxwriter.h>
#endif

#if defined(TECA_HAS_NETCDF)
#include <netcdf.h>
#include "teca_netcdf_util.h"
#endif

using namespace teca_variant_array_util;

namespace internal
{
// ********************************************************************************
int write_csv(const_p_teca_table table, const std::string &file_name)
{
    std::ofstream os(file_name.c_str());
    if (!os.good())
    {
        const char *estr = strerror(errno);
        TECA_ERROR("Failed to open \"" << file_name << "\" for writing. " << estr)
        return -1;
    }

    if (table->to_stream(os))
    {
        TECA_ERROR("Failed to serialize to stream \"" << file_name << "\"")
        return -1;
    }

    return 0;
}

// ********************************************************************************
int write_bin(const_p_teca_table table, const std::string &file_name)
{
    // serialize the table to a binary representation
    teca_binary_stream bs;
    table->to_stream(bs);

    if (teca_file_util::write_stream(file_name.c_str(),
        S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH, "teca_table", bs))
    {
        TECA_ERROR("Failed to write \"" << file_name << "\"")
        return -1;
    }

    return 0;
}

#if defined(TECA_HAS_NETCDF)
// ********************************************************************************
int write_netcdf(const_p_teca_table table, const std::string &file_name,
		 const std::string &row_dim_name)
{
    // create the file
    teca_netcdf_util::netcdf_handle fh;
    if (fh.create(file_name, NC_CLOBBER))
    {
        TECA_ERROR("Failed to create \"" << file_name << "\"")
        return -1;
    }

    // define the row dimension
    int ierr = 0;
    int row_dim_id = 0;
    size_t n_rows = table->get_number_of_rows();
#if !defined(HDF5_THREAD_SAFE)
    {
    std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
    if ((ierr = nc_def_dim(fh.get(), row_dim_name.c_str(), n_rows, &row_dim_id)) != NC_NOERR)
    {
        TECA_ERROR("Failed to create n_rows dimension. " << nc_strerror(ierr))
        return -1;
    }
#if !defined(HDF5_THREAD_SAFE)
    }
#endif

    // define the variables
    std::map<std::string, int> var_ids;

    int n_cols = table->get_number_of_columns();
    for (int i = 0; i < n_cols; ++i)
    {
        std::string col_name = table->get_column_name(i);
        const_p_teca_variant_array col = table->get_column(i);

        int var_id = 0;
#if !defined(HDF5_THREAD_SAFE)
        {
        std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
        VARIANT_ARRAY_DISPATCH(
            col.get(),
            if ((ierr = nc_def_var(fh.get(), col_name.c_str(),
                teca_netcdf_util::netcdf_tt<NT>::type_code, 1,
                &row_dim_id, &var_id)) != NC_NOERR)
            {
                TECA_ERROR("Failed to create numeric variable for column "
                    << i << " \"" << col_name << "\"")
                return -1;
            }
            )
        else VARIANT_ARRAY_DISPATCH_CASE(
            std::string, col.get(),
            if ((ierr = nc_def_var(fh.get(), col_name.c_str(),
                NC_STRING, 1, &row_dim_id, &var_id)) != NC_NOERR)
            {
                TECA_ERROR("Failed to create string variable for column "
                    << i << " \"" << col_name << "\"")
                return -1;
            }
            )
        else
        {
            TECA_ERROR("Failed to create variable for column "
                << i << " \"" << col_name << "\" of type "
                << col->get_class_name())
        }
#if !defined(HDF5_THREAD_SAFE)
        }
#endif
        var_ids[col_name] = var_id;
    }

    // pass the attributes
    teca_metadata atrs;
    table->get_metadata().get("attributes", atrs);
    std::map<std::string, int>::iterator it = var_ids.begin();
    std::map<std::string, int>::iterator end = var_ids.end();
    for (; it != end; ++it)
    {
        const std::string &col_name  = it->first;
        int col_var_id = it->second;

        teca_metadata col_atts;
        if (atrs.get(col_name, col_atts) == 0)
        {
            if (teca_netcdf_util::write_variable_attributes(
                fh, col_var_id, col_atts))
            {
                TECA_ERROR("Failed to write the attributes for column \""
                    << col_name << "\"")
            }
        }
    }

    // put the file into write mode
    nc_enddef(fh.get());

    // write the columns
    for (int i = 0; i < n_cols; ++i)
    {
        std::string col_name = table->get_column_name(i);
        const_p_teca_variant_array col = table->get_column(i);
        int var_id = var_ids[col_name];

        size_t starts = 0;
        size_t counts = n_rows;

        VARIANT_ARRAY_DISPATCH(
            col.get(),
            auto [sp_col, p_col] = get_host_accessible<CTT>(col);
            sync_host_access_any(col);
#if !defined(HDF5_THREAD_SAFE)
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
            if ((ierr = nc_put_vara(fh.get(), var_id, &starts, &counts, p_col)) != NC_NOERR)
            {
                TECA_ERROR("failed to write numeric column " << i << " \""
                    << col_name << "\". " << nc_strerror(ierr))
                return -1;
            }
#if !defined(HDF5_THREAD_SAFE)
            }
#endif
            )
        else VARIANT_ARRAY_DISPATCH_CASE(
            std::string, col.get(),
            auto [sp_col, p_col] = get_host_accessible<CTT>(col);
            sync_host_access_any(col);
            // put the strings into a buffer for netcdf
            const char **string_data = (const char **)malloc(n_rows*sizeof(char*));
            for (size_t j = 0; j < n_rows; ++j)
            {
                string_data[j] = p_col[j].c_str();
            }
#if !defined(HDF5_THREAD_SAFE)
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
            if ((ierr = nc_put_vara(fh.get(), var_id, &starts, &counts, string_data)) != NC_NOERR)
            {
                free(string_data);
                TECA_ERROR("failed to write string column " << i << " \""
                    << col_name << "\". " << nc_strerror(ierr))
                return -1;
            }
#if !defined(HDF5_THREAD_SAFE)
            }
#endif
            free(string_data);
            )
        else
        {
            TECA_ERROR("Failed to write column " << i << " \"" << col_name
                << "\" of type " << col->get_class_name())
        }
    }

    // finish up
    fh.close();

    return 0;
}
#endif

#if defined(TECA_HAS_LIBXLSXWRITER)
// ********************************************************************************
int write_xlsx(const_p_teca_table table, lxw_worksheet *worksheet)
{
    // write column headers
    unsigned int n_cols = table->get_number_of_columns();
    for (unsigned int i = 0; i < n_cols; ++i)
        worksheet_write_string(worksheet, 0, i, table->get_column_name(i).c_str(), NULL);

    // write the columns
    unsigned long long n_rows = table->get_number_of_rows();
    for (unsigned long long j = 0; j < n_rows; ++j)
    {
        for (unsigned int i = 0; i < n_cols; ++i)
        {
            VARIANT_ARRAY_DISPATCH(
                table->get_column(i).get(),
                const TT *a = dynamic_cast<const TT*>(table->get_column(i).get());
                NT v = NT();
                a->get(j, v);
                worksheet_write_number(worksheet, j+1, i, static_cast<double>(v), NULL);
                )
            else VARIANT_ARRAY_DISPATCH_CASE(
                std::string,
                table->get_column(i).get(),
                const TT *a = dynamic_cast<const TT*>(table->get_column(i).get());
                NT v = NT();
                a->get(j, v);
                worksheet_write_string(worksheet, j+1, i, v.c_str(), NULL);
                )
        }
    }

    return 0;
}
#endif
};


// --------------------------------------------------------------------------
teca_table_writer::teca_table_writer() :
     file_name("table_%t%.bin"), row_dim_name("n_rows"),
     output_format(format_auto)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_table_writer::~teca_table_writer()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_table_writer::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_table_writer":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, file_name,
            "path/name of file to write")
        TECA_POPTS_GET(std::string, prefix, row_dim_name,
            "name of the row dimension (only used if output format is netCDF)")
        TECA_POPTS_GET(int, prefix, output_format,
            "output file format enum, 0:csv, 1:bin, 2:xlsx, 3:auto."
            "if auto is used, format is deduced from file_name")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_table_writer::set_properties(const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::string, prefix, file_name)
    TECA_POPTS_SET(opts, std::string, prefix, row_dim_name)
    TECA_POPTS_SET(opts, bool, prefix, output_format)
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_table_writer::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_table_writer::get_output_metadata" << std::endl;
#endif
    (void)port;

    const teca_metadata &md = input_md[0];
    return md;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_table_writer::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
    (void) port;
    (void) request;

    // in parallel only rank 0 is required to have data
    int rank = 0;
#if defined(TECA_HAS_MPI)
    int init = 0;
    MPI_Initialized(&init);
    if (init)
        MPI_Comm_rank(this->get_communicator(), &rank);
#endif
    if (!input_data[0])
    {
        if (rank == 0)
        {
            TECA_FATAL_ERROR("empty input")
        }
        return nullptr;
    }

    // get the current index
    unsigned long index = 0;
    if (input_data[0]->get_request_index(index))
    {
        TECA_FATAL_ERROR("Failed to get the request index of the input data")
        return nullptr;
    }

    // replace time step
    std::string out_file = this->file_name;
    teca_file_util::replace_timestep(out_file, index);

    // replace extension
    int fmt = this->output_format;
    if (fmt == format_auto)
    {
        if (out_file.rfind(".bin") != std::string::npos)
        {
            fmt = format_bin;
        }
        else if (out_file.rfind(".csv") != std::string::npos)
        {
            fmt = format_csv;
        }
        else if (out_file.rfind(".nc") != std::string::npos)
        {
            fmt = format_netcdf;
        }
        else if (out_file.rfind(".xlsx") != std::string::npos)
        {
            fmt = format_xlsx;
        }
        else
        {
            if (rank == 0)
            {
                TECA_WARNING("Failed to determine extension from file name \""
                    << out_file << "\". Using bin format.")
            }
            fmt = format_bin;
        }
    }
    else
    {
        const char *ext;
        switch (fmt)
        {
            case format_bin:
                ext = "bin";
                break;
            case format_csv:
                ext = "csv";
                break;
            case format_netcdf:
                ext = "nc";
                break;
            case format_xlsx:
                ext = "xlsx";
                break;
            default:
                TECA_FATAL_ERROR("Invalid output format")
                return nullptr;
        }
        teca_file_util::replace_extension(out_file, ext);
    }

    // convert table to database
    const_p_teca_table table
        = std::dynamic_pointer_cast<const teca_table>(input_data[0]);

    const_p_teca_database database;

    if (table)
    {
        p_teca_database tmp = teca_database::New();
        tmp->append_table("table 1",
            std::const_pointer_cast<teca_table>(table));
        database = tmp;
    }
    else
    {
        database = std::dynamic_pointer_cast
            <const teca_database>(input_data[0]);
        if (!database)
        {
            TECA_FATAL_ERROR("input must be a table or a database")
            return nullptr;
        }
    }

    // write csv
    if (fmt == format_csv)
    {
        unsigned int n = database->get_number_of_tables();
        for (unsigned int i = 0; i < n; ++i)
        {
            std::string name = database->get_table_name(i);
            std::string out_file_i = out_file;
            teca_file_util::replace_identifier(out_file_i, name);
            const_p_teca_table table = database->get_table(i);
            if (internal::write_csv(table, out_file_i))
            {
                TECA_FATAL_ERROR("Failed to write table " << i << " \"" << name << "\"")
                return nullptr;
            }
        }
    }
    // write binary
    else if (fmt == format_bin)
    {
        unsigned int n = database->get_number_of_tables();
        for (unsigned int i = 0; i < n; ++i)
        {
            std::string name = database->get_table_name(i);
            std::string out_file_i = out_file;
            teca_file_util::replace_identifier(out_file_i, name);
            const_p_teca_table table = database->get_table(i);
            if (internal::write_bin(table, out_file_i))
            {
                TECA_FATAL_ERROR("Failed to write table " << i << " \"" << name << "\"")
                return nullptr;
            }
        }
    }
    // write netcdf
    else if (fmt == format_netcdf)
    {
#if defined(TECA_HAS_NETCDF)
        unsigned int n = database->get_number_of_tables();
        for (unsigned int i = 0; i < n; ++i)
        {
            std::string name = database->get_table_name(i);
            std::string out_file_i = out_file;
            teca_file_util::replace_identifier(out_file_i, name);
            const_p_teca_table table = database->get_table(i);
            if (internal::write_netcdf(table, out_file_i, this->row_dim_name))
            {
                TECA_FATAL_ERROR("Failed to write table " << i << " \"" << name << "\"")
                return nullptr;
            }
        }
#else
        TECA_FATAL_ERROR("Can't write table in NetCDF format because TECA "
            "was not compiled with NetCDF support enabled")
        return nullptr;
#endif
    }
    else if (fmt == format_xlsx)
    {
#if defined(TECA_HAS_LIBXLSXWRITER)
        // open the workbook
        lxw_workbook_options options;
        options.constant_memory = 1;

        lxw_workbook *workbook  =
            workbook_new_opt(out_file.c_str(), &options);

        if (!workbook)
        {
            TECA_FATAL_ERROR("xlsx failed to create workbook ")
        }

        unsigned int n = database->get_number_of_tables();
        for (unsigned int i = 0; i < n; ++i)
        {
            // add a sheet for the table
            std::string name = database->get_table_name(i);
            lxw_worksheet *worksheet =
                workbook_add_worksheet(workbook, name.c_str());

            if (internal::write_xlsx(database->get_table(i), worksheet))
            {
                TECA_FATAL_ERROR("Failed to write table " << i << " \"" << name << "\"")
                return nullptr;
            }
        }

        // close the workbook
        workbook_close(workbook);
#else
        TECA_FATAL_ERROR("Can't write table in MS Excel xlsx format because TECA "
            "was not compiled with xlsx support enabled")
        return nullptr;
#endif
    }
    else
    {
        TECA_FATAL_ERROR("invalid output format " << fmt)
        return nullptr;
    }

    // pass the output through
    p_teca_dataset output = input_data[0]->new_instance();
    output->shallow_copy(std::const_pointer_cast<teca_dataset>(input_data[0]));
    return output;
}
