#include "teca_table_to_stream.h"

#include "teca_table.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <set>
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

using std::string;
using std::vector;
using std::set;
using std::cerr;
using std::endl;

//#define TECA_DEBUG

// --------------------------------------------------------------------------
teca_table_to_stream::teca_table_to_stream() : stream(&std::cerr)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_table_to_stream::~teca_table_to_stream()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_table_to_stream::get_properties_description(
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_table_to_stream":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, header,
            "text to precede table output")
        TECA_POPTS_GET(std::string, prefix, footer,
            "text to follow table output")
        TECA_POPTS_GET(std::string, prefix, stream,
            "name of stream to send output to. stderr, stdout")
        ;
    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_table_to_stream::set_properties(
    const string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, std::string, prefix, header)
    TECA_POPTS_SET(opts, std::string, prefix, footer)
    TECA_POPTS_SET(opts, std::string, prefix, stream)
}
#endif

// --------------------------------------------------------------------------
void teca_table_to_stream::set_stream(std::ostream &s)
{
    this->stream = &s;
}

// --------------------------------------------------------------------------
void teca_table_to_stream::set_stream(const std::string &s)
{
    if ((s == "stderr") || (s == "err") || (s == "cerr") || (s == "std::cerr"))
    {
        this->set_stream_to_stderr();
    }
    else
    if ((s == "stdout") || (s == "out") || (s == "cout") || (s == "std::cout"))
    {
        this->set_stream_to_stdout();
    }
    else
    {
        TECA_ERROR("unknown stream requested \"" << s << "\"")
    }
}

// --------------------------------------------------------------------------
void teca_table_to_stream::set_stream_to_stderr()
{
    this->stream = &std::cerr;
}

// --------------------------------------------------------------------------
void teca_table_to_stream::set_stream_to_stdout()
{
    this->stream = &std::cout;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_table_to_stream::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "teca_table_to_stream::execute" << endl;
#endif
    (void)port;
    (void)request;

    if (!this->stream)
    {
        TECA_ERROR("output stream not set")
        return nullptr;
    }

    // get the input
    const_p_teca_table in_table
        = std::dynamic_pointer_cast<const teca_table>(input_data[0]);

    // in parallel only rank 0 is required to have data
    int rank = 0;
#if defined(TECA_HAS_MPI)
    int init = 0;
    MPI_Initialized(&init);
    if (init)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    if (!in_table)
    {
        if (rank == 0)
        {
            TECA_ERROR("empty input")
        }
        return nullptr;
    }

    // pass the data through so this can sit anywhere in the pipeline
    p_teca_table out_table = teca_table::New();
    out_table->shallow_copy(std::const_pointer_cast<teca_table>(in_table));

    if (!this->header.empty())
        *this->stream << this->header << std::endl;
    out_table->to_stream(*this->stream);
    if (!this->footer.empty())
        *this->stream << this->footer << std::endl;

    return out_table;
}
