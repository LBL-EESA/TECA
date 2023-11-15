#ifndef teca_shape_file_mask_h
#define teca_shape_file_mask_h

#include "teca_config.h"
#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_shape_file_mask)

/// Generates a valid value mask defined by regions in the given ESRI shape file
/** The teca_shape_file_mask generates a mask on a 2D mesh where mesh points
 * are assigned a value of 1 where they fall inside any one of the polygons
 * defined in the named ESRI shape file, and 0 everywhere else. The input mesh
 * is passed through to the output with the mask added. By default a transform
 * on the x coordinates of the polygons from [-180, 180] to [0, 360] is
 * applied.  If this is undesirable it can be disabled by setting the
 * normalize_coordinates property to 0.
 *
 * @attention Currently there is no check on the coordinate system specified in
 * the shapefile. This could result in a mask of all zeros when the shapefile
 * and input mesh use a different coodinate system.
 */
class TECA_EXPORT teca_shape_file_mask : public teca_algorithm
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

    /** @name normalize_coordinates
     * set this flag to transform the x coordinates from [-180, 180] to
     * [0, 360]
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, normalize_coordinates)
    ///@}

    /** @name number_of_cuda_streams
     * sets the number of streams to use when searching the mesh for polygon
     * intersections.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, number_of_cuda_streams)
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
    int normalize_coordinates;
    int number_of_cuda_streams;

    struct internals_t;
    internals_t *internals;
};

#endif
