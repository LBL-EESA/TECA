#ifndef array_add_h
#define array_add_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(array_add)

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
        const teca_metadata &input_md,
        std::string &active_array) const;

private:
    teca_metadata get_output_metadata(
        unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port,
        const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::string array_1;
    std::string array_2;
};

#endif

