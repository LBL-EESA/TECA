#ifndef teca_index_executive_h
#define teca_index_executive_h

#include "teca_shared_object.h"
#include "teca_algorithm_executive.h"
#include "teca_metadata.h"

#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_index_executive)

///
/**
An executive that generates requests using a upstream
or user defined index. an extent and list of arrays
can be optionally set.

meta data keys:

     requires:

     index_initializer_key -- holds the name of the key that tells how
                              many indices are available. the named key
                              must also be present and should conatin the
                              number of indices available

     index_request_key -- holds the name of the key used to request
                          a specific index. request are generated with this
                          name set to a specific index to be processed some
                          upstream algorithm is expected to produce the data
                          associated with the given index

*/
class teca_index_executive : public teca_algorithm_executive
{
public:
    TECA_ALGORITHM_EXECUTIVE_STATIC_NEW(teca_index_executive)

    virtual int initialize(const teca_metadata &md);
    virtual teca_metadata get_next_request();

    // set the index to process
    void set_index(long s);

    // set the first time step in the series to process.
    // default is 0.
    void set_start_index(long s);

    // set the last time step in the series to process.
    // default is -1. negative number results in the last
    // available time step being used.
    void set_end_index(long s);

    // set the stride to process time steps at. default
    // is 1
    void set_stride(long s);

    // set the extent to process. the default is the
    // whole_extent.
    void set_extent(unsigned long *ext);
    void set_extent(const std::vector<unsigned long> &ext);

    // set the list of arrays to process
    void set_arrays(const std::vector<std::string> &arrays);

protected:
    teca_index_executive();

private:
    std::vector<teca_metadata> requests;
    std::string index_initializer_key;
    std::string index_request_key;
    long start_index;
    long end_index;
    long stride;
    std::vector<unsigned long> extent;
    std::vector<std::string> arrays;
};

#endif
