#include "teca_csv_writer.h"

#include "teca_table.h"
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

using std::vector;
using std::string;
using std::ostringstream;
using std::ofstream;
using std::cerr;
using std::endl;

// --------------------------------------------------------------------------
teca_csv_writer::teca_csv_writer()
    : file_name("table_%t%.%e%"), binary_mode(false)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_csv_writer::~teca_csv_writer()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_csv_writer::get_properties_description(
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for " + prefix + "(teca_csv_writer)");

    opts.add_options()
        TECA_POPTS_GET(string, prefix, file_name, "path/name of file to write")
        TECA_POPTS_GET(bool, prefix, binary_mode, "write binary")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_csv_writer::set_properties(const string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, string, prefix, file_name)
    TECA_POPTS_SET(opts, bool, prefix, binary_mode)
}
#endif

// --------------------------------------------------------------------------
const_p_teca_dataset teca_csv_writer::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
    (void) port;

    // handle the case that no data is sent. it is not
    // an error.
    if (!input_data[0])
        return nullptr;

    // get the input
    const_p_teca_table table
        = std::dynamic_pointer_cast<const teca_table>(input_data[0]);

    if (!table)
    {
        TECA_ERROR("input dataset is not a teca_table")
        return nullptr;
    }

    if (!table->empty())
    {
        string out_file = this->file_name;

        // replace time step
        unsigned long time_step = 0l;
        request.get("time_step", time_step);
        teca_file_util::replace_timestep(out_file, time_step);

        // replace extension
        string ext;
        if (this->binary_mode)
            ext = "bin";
        else
            ext = "csv";
        teca_file_util::replace_extension(out_file, ext);

        // write the data
        if (this->binary_mode)
        {
            if (this->write_bin(table, out_file))
            {
                TECA_ERROR("Failed to write binary file \"" << out_file << "\"")
                return nullptr;
            }
        }
        else
        {
            if (this->write_csv(table, out_file))
            {
                TECA_ERROR("Failed to write csv file \"" << out_file << "\"")
                return nullptr;
            }
        }
    }

    return nullptr;
}

// --------------------------------------------------------------------------
int teca_csv_writer::write_csv(
    const_p_teca_table table,
    const std::string &file_name)
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

// --------------------------------------------------------------------------
int teca_csv_writer::write_bin(
    const_p_teca_table table,
    const std::string &file_name)
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
