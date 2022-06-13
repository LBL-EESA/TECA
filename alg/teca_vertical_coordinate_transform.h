#ifndef teca_vertical_coordinate_transform_h
#define teca_vertical_coordinate_transform_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_vertical_coordinate_transform)

/// An algorithm that transforms the vertical cooridinates of a mesh
class TECA_EXPORT teca_vertical_coordinate_transform : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_vertical_coordinate_transform)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_vertical_coordinate_transform)
    TECA_ALGORITHM_CLASS_NAME(teca_vertical_coordinate_transform)
    ~teca_vertical_coordinate_transform();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()


    // set the transform mode.
    enum {
        mode_invalid = 0,
        mode_wrf_v3 = 1
    };
    TECA_ALGORITHM_PROPERTY(int, mode)

protected:
    teca_vertical_coordinate_transform();

private:
    using teca_algorithm::get_output_metadata;

    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    int mode;
};

#endif
