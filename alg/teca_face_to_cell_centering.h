#ifndef teca_face_to_cell_centering_h
#define teca_face_to_cell_centering_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_face_to_cell_centering)

/// An  algorithm that transforms from face to cell centering
class TECA_EXPORT teca_face_to_cell_centering : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_face_to_cell_centering)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_face_to_cell_centering)
    TECA_ALGORITHM_CLASS_NAME(teca_face_to_cell_centering)
    ~teca_face_to_cell_centering();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

protected:
    teca_face_to_cell_centering();

private:
    using teca_algorithm::get_output_metadata;

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
};

#endif
