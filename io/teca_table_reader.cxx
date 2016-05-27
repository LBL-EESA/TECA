#include "teca_table_reader.h"
#include "teca_table.h"

#include <algorithm>
#include <cstring>
#include <cstdio>

using std::string;
using std::endl;
using std::cerr;

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

// --------------------------------------------------------------------------
teca_table_reader::teca_table_reader()
{}

// --------------------------------------------------------------------------
teca_table_reader::~teca_table_reader()
{
}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_table_reader::get_properties_description(
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_table_reader":prefix));

    opts.add_options()
        TECA_POPTS_GET(string, prefix, file_name, "a file name to read")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_table_reader::set_properties(const string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, string, prefix, file_name)
}
#endif

// --------------------------------------------------------------------------
const_p_teca_dataset teca_table_reader::execute(
    unsigned int,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
    (void)input_data;
    (void)request;

    // in parallel only rank 0 will read the data
#if defined(TECA_HAS_MPI)
    int rank = 0;
    int init = 0;
    MPI_Initialized(&init);
    if (init)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0)
        return nullptr;
#endif

    // open the file
    teca_binary_stream bs;
    FILE* fd = fopen(file_name.c_str(), "rb");
    if (fd == NULL)
    {
        TECA_ERROR("Failed to open " << file_name << endl)
        return nullptr;
    }

    // get its length, we'll read it in one go and need to create
    // a bufffer for it's contents
    long start = ftell(fd);
    fseek(fd, 0, SEEK_END);
    long end = ftell(fd);
    fseek(fd, 0, SEEK_SET);
    long nbytes = end - start - 10;

    // check if this is really ours
    char id[11] = {'\0'};
    if (fread(id, 1, 10, fd) != 10)
    {
        const char *estr = (ferror(fd) ? strerror(errno) : "");
        fclose(fd);
        TECA_ERROR("Failed to read \"" << file_name << "\". " << estr)
        return nullptr;
    }

    if (strncmp(id, "teca_table", 10))
    {
        fclose(fd);
        TECA_ERROR("Not a teca_table. \"" << file_name << "\"")
        return nullptr;
    }

    // create the buffer
    bs.resize(static_cast<size_t>(nbytes));

    // read the stream
    long bytes_read = fread(bs.get_data(), sizeof(unsigned char), nbytes, fd);
    if (bytes_read != nbytes)
    {
        const char *estr = (ferror(fd) ? strerror(errno) : "");
        fclose(fd);
        TECA_ERROR("Failed to read \"" << file_name << "\". Read only "
            << bytes_read << " of the requested " << nbytes << ". " << estr)
        return nullptr;
    }
    fclose(fd);

    // deserialize the binary rep
    p_teca_table table = teca_table::New();
    table->from_stream(bs);
    return table;
}
