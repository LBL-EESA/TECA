#ifndef array_writer_h
#define array_writer_h

#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <memory>
#include <vector>
#include <iostream>

class array_writer;
typedef std::shared_ptr<array_writer> p_array_writer;

/**
an example implementation of a teca_algorithm
that writes arrays to a user provided stream
*/
class array_writer : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(array_writer)
    ~array_writer();

protected:
    array_writer();

private:
    virtual
    p_teca_dataset execute(
        unsigned int port,
        const std::vector<p_teca_dataset> &input_data,
        const teca_metadata &request);
};

#endif
