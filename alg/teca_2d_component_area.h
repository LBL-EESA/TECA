#ifndef teca_2d_component_area_h
#define teca_2d_component_area_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_2d_component_area)

/// an algorithm that computes the area of labeled regions
/**
Given a set of labels on a Cartesian mesh, the algorithm computes
the area of each region. Regions are identified by assigning a
unique integer value to each mesh point that belongs in the
region. The component_variable property names the variable containing
the region labels.

if the region labels start at 0 and are contiguous then an
optimization can be used. Set contiguous_component_ids property
to enable the optimization.

the input dataset is passed through and the results of the
calculations are stored in the output dataset metadata in
keys named:

  number_of_components
  component_ids
  component_area

*/
class teca_2d_component_area : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_2d_component_area)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_2d_component_area)
    TECA_ALGORITHM_CLASS_NAME(teca_2d_component_area)
    ~teca_2d_component_area();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set the name of the input array
    TECA_ALGORITHM_PROPERTY(std::string, component_variable)

    // set this only if you know for certain that label ids
    // are contiguous and start at 0. this enables use of a
    // faster implementation.
    TECA_ALGORITHM_PROPERTY(int, contiguous_component_ids)

protected:
    teca_2d_component_area();

    int get_component_variable(std::string &component_var);

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
    std::string component_variable;
    int contiguous_component_ids;
};

#endif
