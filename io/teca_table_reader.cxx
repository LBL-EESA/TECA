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
    options_description opts("Options for " + prefix + "(teca_table_reader)");

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

    // Open the file and dump its contents into a binary stream.
    teca_binary_stream bs;
    FILE* fd = fopen(file_name.c_str(), "rb");
    long start = ftell(fd);
    fseek(fd, 0, SEEK_END);
    long end = ftell(fd);
    fseek(fd, 0, SEEK_SET);
    bs.resize(static_cast<size_t>(end - start));
    fread(bs.get_data(), sizeof(unsigned char), end - start, fd);
    fclose(fd);

    // Read table data from the stream.
    p_teca_table table = teca_table::New();
    table->from_stream(bs);
    return table;
}
