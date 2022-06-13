#include "teca_cf_time_axis_data_reduce.h"
#include "teca_cf_time_axis_data.h"

#include <iostream>
#include <limits>

using std::cerr;
using std::endl;
using std::vector;

// --------------------------------------------------------------------------
teca_cf_time_axis_data_reduce::teca_cf_time_axis_data_reduce()
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_cf_time_axis_data_reduce::initialize_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_cf_time_axis_data_reduce::initialize_upstream_request" << endl;
#endif
    (void) port;
    (void) input_md;

    teca_metadata req(request);
    req.remove("index_request_key");
    req.remove("axis");

    vector<teca_metadata> up_reqs(1, req);
    return up_reqs;
}

// --------------------------------------------------------------------------
teca_metadata teca_cf_time_axis_data_reduce::initialize_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_cf_time_axis_data_reduce::intialize_output_metadata" << endl;
#endif
    (void) port;
    (void) input_md;

    teca_metadata output_md;
    output_md.set("index_initializer_key", std::string("num_axes"));
    output_md.set("num_axes", int(1));
    output_md.set("index_request_key", std::string("axis"));

    return output_md;
}

// --------------------------------------------------------------------------
p_teca_dataset teca_cf_time_axis_data_reduce::reduce(int device_id,
    const const_p_teca_dataset &l_ds, const const_p_teca_dataset &r_ds)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_cf_time_axis_data_reduce::reduce" << endl;
#endif
    (void) device_id;

    p_teca_cf_time_axis_data l_axis_data =
        std::dynamic_pointer_cast<teca_cf_time_axis_data>
            (std::const_pointer_cast<teca_dataset>(l_ds));

    p_teca_cf_time_axis_data r_axis_data =
        std::dynamic_pointer_cast<teca_cf_time_axis_data>
            (std::const_pointer_cast<teca_dataset>(r_ds));

    p_teca_cf_time_axis_data output_axis_data;

    bool have_l = l_axis_data && *l_axis_data;
    bool have_r = r_axis_data && *r_axis_data;

    if (have_l && have_r)
    {
        output_axis_data =
            std::dynamic_pointer_cast<teca_cf_time_axis_data>
                (l_axis_data->new_shallow_copy());

        output_axis_data->shallow_append(r_axis_data);
    }
    else
    if (have_l)
    {
        output_axis_data =
            std::dynamic_pointer_cast<teca_cf_time_axis_data>
                (l_axis_data->new_shallow_copy());
    }
    else
    if (have_r)
    {
        output_axis_data =
            std::dynamic_pointer_cast<teca_cf_time_axis_data>
                (r_axis_data->new_shallow_copy());
    }

    return output_axis_data;
}
