#include "teca_table_writer.h"

#include "teca_table.h"
#include "teca_database.h"
#include "teca_metadata.h"
#include "teca_binary_stream.h"
#include "teca_file_util.h"

#include <iostream>
#include <sstream>
#include <fstream>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

#if defined(TECA_HAS_LIBXLSXWRITER)
#include <xlsxwriter.h>
#endif

using std::vector;
using std::string;
using std::ostringstream;
using std::ofstream;
using std::cerr;
using std::endl;

namespace internal
{
// ********************************************************************************
int write_csv(const_p_teca_table table, const std::string &file_name)
{
    ofstream os(file_name.c_str());
    if (!os.good())
    {
        TECA_ERROR("Failed to open \"" << file_name << "\" for writing")
        return -1;
    }

    table->to_stream(os);

    return 0;
}

// ********************************************************************************
int write_bin(const_p_teca_table table, const std::string &file_name)
{
    // serialize the table to a binary representation
    teca_binary_stream bs;
    table->to_stream(bs);

    if (teca_file_util::write_stream(file_name.c_str(), "teca_table", bs))
    {
        TECA_ERROR("Failed to write \"" << file_name << "\"")
        return -1;
    }

    return 0;
}

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
            TEMPLATE_DISPATCH(const teca_variant_array_impl,
                table->get_column(i).get(),
                const TT *a = dynamic_cast<const TT*>(table->get_column(i).get());
                NT v = NT();
                a->get(j, v);
                worksheet_write_number(worksheet, j+1, i, static_cast<double>(v), NULL);
                )
            else TEMPLATE_DISPATCH_CASE(const teca_variant_array_impl,
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
teca_table_writer::teca_table_writer()
    : file_name("table_%t%.bin"), output_format(format_auto)
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
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_table_writer":prefix));

    opts.add_options()
        TECA_POPTS_GET(string, prefix, file_name,
            "path/name of file to write")
        TECA_POPTS_GET(int, prefix, output_format,
            "output file format enum, 0:csv, 1:bin, 2:xlsx, 3:auto."
            "if auto is used, format is deduced from file_name")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_table_writer::set_properties(const string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, string, prefix, file_name)
    TECA_POPTS_SET(opts, bool, prefix, output_format)
}
#endif

// --------------------------------------------------------------------------
const_p_teca_dataset teca_table_writer::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
    (void) port;

    // in parallel only rank 0 is required to have data
    int rank = 0;
#if defined(TECA_HAS_MPI)
    int init = 0;
    MPI_Initialized(&init);
    if (init)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    if (!input_data[0])
    {
        if (rank == 0)
        {
            TECA_ERROR("empty input")
        }
        return nullptr;
    }

    string out_file = this->file_name;

    // replace time step
    unsigned long time_step = 0l;
    request.get("time_step", time_step);
    teca_file_util::replace_timestep(out_file, time_step);

    // replace extension
    int fmt = this->output_format;
    if (fmt == format_auto)
    {
        if (out_file.rfind(".xlsx") != std::string::npos)
        {
            fmt = format_xlsx;
        }
        else if (out_file.rfind(".csv") != std::string::npos)
        {
            fmt = format_csv;
        }
        else if (out_file.rfind(".bin") != std::string::npos)
        {
            fmt = format_bin;
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
            case format_xlsx:
                ext = "xlsx";
                break;
            default:
                TECA_ERROR("Invalid output format")
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
            TECA_ERROR("input must be a table or a database")
            return nullptr;
        }
    }

    // write based on format
    switch (fmt)
    {
        case format_csv:
        case format_bin:
            {
            unsigned int n = database->get_number_of_tables();
            for (unsigned int i = 0; i < n; ++i)
            {
                std::string name = database->get_table_name(i);
                std::string out_file_i = out_file;
                teca_file_util::replace_identifier(out_file_i, name);
                const_p_teca_table table = database->get_table(i);
                if (((fmt == format_csv) && internal::write_csv(table, out_file_i))
                  || ((fmt == format_bin) && internal::write_bin(table, out_file_i)))
                {
                    TECA_ERROR("Failed to write table " << i << " \"" << name << "\"")
                    return nullptr;
                }
            }
            }
            break;
        case format_xlsx:
            {
#if defined(TECA_HAS_LIBXLSXWRITER)
            // open the workbook
            lxw_workbook_options options;
            options.constant_memory = 1;

            lxw_workbook *workbook  =
                workbook_new_opt(out_file.c_str(), &options);

            if (!workbook)
            {
                TECA_ERROR("xlsx failed to create workbook ")
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
                    TECA_ERROR("Failed to write table " << i << " \"" << name << "\"")
                    return nullptr;
                }
            }

            // close the workbook
            workbook_close(workbook);
#else
            TECA_ERROR("TECA was not compiled with libxlsx support")
#endif
            }
            break;
        default:
            TECA_ERROR("invalid output format")
    }

    // pass the output through
    p_teca_dataset output = input_data[0]->new_instance();
    output->shallow_copy(std::const_pointer_cast<teca_dataset>(input_data[0]));
    return output;
}
