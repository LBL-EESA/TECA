#ifndef teca_descriptive_statistics_h
#define teca_descriptive_statistics_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_descriptive_statistics)

/// compute descriptive statistics over a set of arrays.
/**
compute the min, max, avg, median, standard deviation of a
set of named arrays. the results are returned in a table.
*/
class teca_descriptive_statistics : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_descriptive_statistics)
    ~teca_descriptive_statistics();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set/get the list of arrays that are needed to produce
    // the derived quantity
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, dependent_variable)

protected:
    teca_descriptive_statistics();

private:
    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(
        unsigned int port, const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

    void get_dependent_variables(const teca_metadata &request,
        std::vector<std::string> &dep_vars);
private:
    std::vector<std::string> dependent_variables;
};

#endif
