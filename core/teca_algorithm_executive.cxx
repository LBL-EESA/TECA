#include "teca_algorithm_executive.h"

// --------------------------------------------------------------------------
int teca_algorithm_executive::initialize(const teca_metadata &)
{
    // make a non-empty request. any key that's not used will
    // work here, prepending __ to add some extra safety in this
    // regard.
    m_md.insert("__request_empty", 0);
    return 0;
}

// --------------------------------------------------------------------------
teca_metadata teca_algorithm_executive::get_next_request()
{
    // send the cached request and replace it with
    // an empty one.
    teca_metadata req = m_md;
    m_md = teca_metadata();
    return req;
}
