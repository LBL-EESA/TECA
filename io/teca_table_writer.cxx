#include "teca_table_writer.h"

#include "teca_table.h"
#include "teca_database.h"
#include "teca_metadata.h"
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

#if defined(TECA_HAS_LIBXLSXWRITER)
#include <xlsxwriter.h>
#endif

using std::vector;
using std::string;
using std::ostringstream;
using std::ofstream;
using std::cerr;
using std::endl;


namespace {

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
    teca_binary_stream bs;

    table->to_stream(bs);

    int fd = creat(file_name.c_str(), S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH);
    if (fd == -1)
    {
        const char *estr = strerror(errno);
        TECA_ERROR("Failed to creat " << file_name << endl << estr)
        return -1;
    }

    ssize_t n_wrote = 0;
    ssize_t n_to_write = bs.size();
    while (n_to_write > 0)
    {
        ssize_t n = write(fd, bs.get_data() + n_wrote, n_to_write);
        if (n == -1)
        {
            const char *estr = strerror(errno);
            TECA_ERROR("Failed to write " << file_name << endl << estr)
            return -1;
        }
        n_wrote += n;
        n_to_write -= n;
    }

    if (close(fd))
    {
        const char *estr = strerror(errno);
        TECA_ERROR("Failed to close " << file_name << endl << estr)
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
    : file_name("table_%t%.%e%"), output_format(csv)
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
    options_description opts("Options for " + prefix + "(teca_table_writer)");

    opts.add_options()
        TECA_POPTS_GET(string, prefix, file_name, "path/name of file to write")
        TECA_POPTS_GET(int, prefix, output_format, "output file format, 0 : csv, 1 : bin, 2 : xlsx")
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

    // handle the case that no data is sent. it is not
    // an error.
    if (!input_data[0])
        return nullptr;

    string out_file = this->file_name;

    // replace time step
    unsigned long time_step = 0l;
    request.get("time_step", time_step);
    teca_file_util::replace_timestep(out_file, time_step);

    // replace extension
    string ext;
    switch (this->output_format)
    {
        case bin:
            ext = "bin";
            break;
        case csv:
            ext = "csv";
            break;
        case xlsx:
            ext = "xlsx";
            break;
        default:
            TECA_ERROR("invalid output format")
            return nullptr;
    }
    teca_file_util::replace_extension(out_file, ext);

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
    switch (this->output_format)
    {
        case csv:
        case bin:
            {
            unsigned int n = database->get_number_of_tables();
            for (unsigned int i = 0; i < n; ++i)
            {
                std::string name = database->get_table_name(i);
                std::string out_file_i = out_file;
                teca_file_util::replace_identifier(out_file_i, name);
                const_p_teca_table table = database->get_table(i);
                if (((this->output_format == csv) && ::write_csv(table, out_file_i))
                  || ((this->output_format == bin) && ::write_bin(table, out_file_i)))
                {
                    TECA_ERROR("Failed to write table " << i << " \"" << name << "\"")
                    return nullptr;
                }
            }
            }
            break;
        case xlsx:
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

                if (::write_xlsx(database->get_table(i), worksheet))
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

   return nullptr;
}
