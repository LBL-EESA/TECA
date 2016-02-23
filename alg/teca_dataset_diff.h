#ifndef teca_dataset_diff_h
#define teca_dataset_diff_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_table_fwd.h"
#include "teca_cartesian_mesh_fwd.h"

#include <vector>
#include <string>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_dataset_diff)

/**
an algorithm that writes cartesian meshes in VTK format.
*/
class teca_dataset_diff : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_dataset_diff)
    ~teca_dataset_diff();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // Tolerance below which two floating-point quantities are considered equal.
    TECA_ALGORITHM_PROPERTY(double, tolerance)

protected:
    teca_dataset_diff();

    int compare_tables(const_p_teca_table table1, const_p_teca_table table2);
    int compare_cartesian_meshes(const_p_teca_cartesian_mesh mesh1, const_p_teca_cartesian_mesh mesh2);

private:

    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    double tolerance;
};

#endif
