#ifndef teca_mesh_layering_h
#define teca_mesh_layering_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_mesh_layering)

/**
Pads the specified scalar field with zeroes or, if specified, pad_value.

note that user specified values take precedence over request keys. When using
request keys be sure to include the variable post-fix.
*/
class teca_mesh_layering : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_mesh_layering)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_mesh_layering)
    TECA_ALGORITHM_CLASS_NAME(teca_mesh_layering)
    ~teca_mesh_layering();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    TECA_ALGORITHM_PROPERTY(size_t, n_layers)

protected:
    teca_mesh_layering();

    int get_variable(std::string &var);

private:
    teca_metadata get_output_metadata(
        unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    unsigned long n_layers;
};

#endif
