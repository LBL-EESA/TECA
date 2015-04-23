#include "teca_table_writer.h"

#include "teca_table.h"
#include "teca_metadata.h"

#include <iostream>
#include <sstream>
#include <fstream>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>


using std::vector;
using std::string;
using std::ostringstream;
using std::ofstream;
using std::cerr;
using std::endl;

// --------------------------------------------------------------------------
teca_table_writer::teca_table_writer() : binary_mode(false)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_table_writer::~teca_table_writer()
{}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_table_writer::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
    (void) port;

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
        ostringstream file_name;
        file_name << this->base_file_name;;

        // if there is a time step, use it in the file name
        unsigned long time_step;
        if (!request.get("time_step", time_step))
            file_name << "_" << time_step;

        if (this->binary_mode)
        {
            file_name << ".bin";
            if (this->write_bin(table, file_name.str()))
            {
                TECA_ERROR("Failed to write binary file \"" << file_name.str() << "\"")
                return nullptr;
            }
        }
        else
        {
            file_name << ".csv";
            if (this->write_csv(table, file_name.str()))
            {
                TECA_ERROR("Failed to write csv file \"" << file_name.str() << "\"")
                return nullptr;
            }
        }
    }

    return nullptr;
}

// --------------------------------------------------------------------------
int teca_table_writer::write_csv(
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
int teca_table_writer::write_bin(
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
