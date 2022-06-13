#ifndef teca_component_statistics_h
#define teca_component_statistics_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_component_statistics)

/// compute statistics about connected components
class TECA_EXPORT teca_component_statistics : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_component_statistics)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_component_statistics)
    TECA_ALGORITHM_CLASS_NAME(teca_component_statistics)
    ~teca_component_statistics();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

protected:
    teca_component_statistics();

private:
    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(
        unsigned int port, const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

    void get_dependent_variables(const teca_metadata &request,
        std::vector<std::string> &dep_vars);
};

#endif
