#ifndef array_add_h
#define array_add_h

#include "teca_algorithm.h"
#include "teca_meta_data.h"

#include <memory>
#include <string>
#include <vector>

class array_add;
typedef std::shared_ptr<array_add> p_array_add;

/**
an example implementation of a teca_algorithm
that adds two arrays on its inputs
*/
class array_add : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(array_add)
    ~array_add();

    // sets the names of the two arrays to add
    TECA_ALGORITHM_PROPERTY(std::string, array_1)
    TECA_ALGORITHM_PROPERTY(std::string, array_2)

protected:
    array_add();

    // helper to get the two names to add or
    // default values if the user has not set
    // these
    int get_active_array(
        const std::string &user_array,
        const teca_meta_data &input_md,
        std::string &active_array);

private:
    virtual
    teca_meta_data get_output_meta_data(
        unsigned int port,
        std::vector<teca_meta_data> &input_md);

    virtual
    std::vector<teca_meta_data> get_upstream_request(
        unsigned int port,
        std::vector<teca_meta_data> &input_md,
        teca_meta_data &request);

    virtual
    p_teca_dataset execute(
        unsigned int port,
        std::vector<p_teca_dataset> &input_data,
        teca_meta_data &request);

private:
    std::string array_1;
    std::string array_2;
};

#endif

