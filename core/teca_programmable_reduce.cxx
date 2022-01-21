#include "teca_programmable_reduce.h"

#include <iostream>
#include <limits>


// --------------------------------------------------------------------------
teca_programmable_reduce::teca_programmable_reduce() :
    reduce_callback(nullptr), request_callback(nullptr),
    report_callback(nullptr)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);

    strncpy(this->class_name, "teca_programmable_reduce",
        sizeof(this->class_name));
}

// --------------------------------------------------------------------------
int teca_programmable_reduce::set_name(const std::string &name)
{
    if (snprintf(this->class_name, sizeof(this->class_name),
        "teca_programmable_reduce(%s)", name.c_str()) >=
        static_cast<int>(sizeof(this->class_name)))
    {
        TECA_ERROR("name is too long for the current buffer size "
            << sizeof(this->class_name))
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata>
    teca_programmable_reduce::initialize_upstream_request(unsigned int port,
    const std::vector<teca_metadata> &input_md, const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_programmable_reduce::initialize_upstream_request" << std::endl;
#endif
    if (!this->request_callback)
    {
        std::vector<teca_metadata> up_reqs(1, request);
        return up_reqs;
    }

    return this->request_callback(port, input_md, request);
}

// --------------------------------------------------------------------------
teca_metadata teca_programmable_reduce::initialize_output_metadata(
    unsigned int port, const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_programmable_reduce::intialize_output_metadata" << std::endl;
#endif
    if (!this->report_callback)
    {
        teca_metadata output_md(input_md[0]);
        return output_md;
    }

    return this->report_callback(port, input_md);
}

// --------------------------------------------------------------------------
p_teca_dataset teca_programmable_reduce::reduce(int device_id,
    const const_p_teca_dataset &left_ds, const const_p_teca_dataset &right_ds)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_programmable_reduce::reduce" << std::endl;
#endif

    if (!this->reduce_callback)
    {
        TECA_FATAL_ERROR("a reduce callback has not been provided")
        return nullptr;
    }

    return this->reduce_callback(device_id, left_ds, right_ds);
}

// --------------------------------------------------------------------------
p_teca_dataset teca_programmable_reduce::finalize(int device_id,
    const const_p_teca_dataset &ds)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_programmable_reduce::finalize" << std::endl;
#endif

    if (!this->finalize_callback)
        return this->teca_index_reduce::finalize(device_id, ds);

    return this->finalize_callback(device_id, ds);
}
