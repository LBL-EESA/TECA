#ifndef cf_reader_driver_h
#define cf_reader_driver_h

#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <memory>
#include <vector>
#include <iostream>

class cf_reader_driver;
using p_cf_reader_driver = std::shared_ptr<cf_reader_driver>;

/**
an example implementation of a teca_algorithm
that driver the cf_reader for testing and validation
*/
class cf_reader_driver : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(cf_reader_driver)
    ~cf_reader_driver();

protected:
    cf_reader_driver();

private:
    virtual
    teca_metadata get_output_metadata(
        unsigned int port,
        const std::vector<teca_metadata> &input_md);

    virtual
    std::vector<teca_metadata> get_upstream_request(
        unsigned int port,
        const std::vector<teca_metadata> &input_md,
        const teca_metadata &request);

    virtual
    p_teca_dataset execute(
        unsigned int port,
        const std::vector<p_teca_dataset> &input_data,
        const teca_metadata &request);
};

#endif
