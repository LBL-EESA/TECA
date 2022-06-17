#include "teca_algorithm_executive.h"

// --------------------------------------------------------------------------
int teca_algorithm_executive::initialize(MPI_Comm comm, const teca_metadata &md)
{
    (void)comm;

    // the default request is made for index 0
    std::string request_key;
    if (md.get("index_request_key", request_key))
    {
        TECA_FATAL_ERROR("No index request key has been specified")
        return -1;
    }

    m_request.set("index_request_key", request_key);
    m_request.set(request_key, {0ul, 0ul});

    // set the default device to execute on
#if defined(TECA_HAS_CUDA)
    int device_id = 0;
#else
    int device_id = -1;
#endif
    m_request.set("device_id", device_id);

    return 0;
}

// --------------------------------------------------------------------------
teca_metadata teca_algorithm_executive::get_next_request()
{
    // send the cached request and replace it with
    // an empty one.
    teca_metadata req = m_request;
    m_request = teca_metadata();
    return req;
}
