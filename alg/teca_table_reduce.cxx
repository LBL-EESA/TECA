#include "teca_table_reduce.h"
#include "teca_table.h"

#include <iostream>
#include <limits>

using std::cerr;
using std::endl;
using std::vector;

// --------------------------------------------------------------------------
teca_table_reduce::teca_table_reduce()
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_table_reduce::initialize_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_table_reduce::initialize_upstream_request" << endl;
#endif
    (void) port;
    (void) input_md;

    vector<teca_metadata> up_reqs(1, request);
    return up_reqs;
}

// --------------------------------------------------------------------------
teca_metadata teca_table_reduce::initialize_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_table_reduce::intialize_output_metadata" << endl;
#endif
    (void) port;

    teca_metadata output_md(input_md[0]);
    return output_md;
}

// --------------------------------------------------------------------------
p_teca_dataset teca_table_reduce::reduce(
    const const_p_teca_dataset &left_ds,
    const const_p_teca_dataset &right_ds)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_table_reduce::reduce" << endl;
#endif
    const_p_teca_table left_table
        = std::dynamic_pointer_cast<const teca_table>(left_ds);

    const_p_teca_table right_table
        = std::dynamic_pointer_cast<const teca_table>(right_ds);

    p_teca_table output_table;

    bool left = left_table && *left_table;
    bool right = right_table && *right_table;

    if (left && right)
    {
        output_table
            = std::dynamic_pointer_cast<teca_table>(left_table->new_copy());

        output_table->concatenate_rows(right_table);
    }
    else
    if (left)
    {
        output_table
            = std::dynamic_pointer_cast<teca_table>(left_table->new_copy());
    }
    else
    if (right)
    {
        output_table
            = std::dynamic_pointer_cast<teca_table>(right_table->new_copy());
    }

    return output_table;
}
