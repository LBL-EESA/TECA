#ifndef teca_shape_file_mask_h
#define teca_shape_file_mask_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_shape_file_mask)

/// Generates a valid value mask defined by regions in the given ESRI shape file
/** This algorithm is a source and has no inputs. A
 */
class teca_shape_file_mask : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_shape_file_mask)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_shape_file_mask)
    TECA_ALGORITHM_CLASS_NAME(teca_shape_file_mask)
    ~teca_shape_file_mask();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name shape_file
     * Set the path to the shape file. This file is read by MPI rank 0 and
     * distributed to the others in parallel runs
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, shape_file)
    ///@}

    /** @name mask_variables
     * set the names of the variables to store the generated mask in
     * each variable will contain a reference to the mask
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, mask_variable)
    ///@}


protected:
    teca_shape_file_mask();

    /// overriden so that cached data is cleared
    void set_modified() override;

    /// generate attributes for the output arrays
    teca_metadata get_mask_array_attributes(unsigned long size);

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

private:
    std::string shape_file;
    std::vector<std::string> mask_variables;

    struct internals_t;
    internals_t *internals;
};

#endif
