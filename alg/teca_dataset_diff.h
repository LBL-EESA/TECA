#ifndef teca_dataset_diff_h
#define teca_dataset_diff_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_table_fwd.h"
#include "teca_cartesian_mesh_fwd.h"
#include "teca_array_collection_fwd.h"

#include <vector>
#include <string>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_dataset_diff)

/// compute the element wise difference between to datasets
/**
a two input algorithm that compares datasets by examining each
element of their contained arrays. a threshold is used to detect
when an element is different. a report containing the string FAIL
is issued to stderr stream when a difference is detected. this
algorithm is the core of TECA's regression test suite.

by convention the first input produces the reference dataset,
and the second input produces the dataset to validate. this is
primarilly to support map-reduce implementation where after
the reduction only rank 0 has data.
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

    // Relative tolerance below which two floating-point quantities are 
    // considered equal. The relative difference for a computed quantity A and 
    // a reference quantity B is
    // rel_diff = |A - B| / B, B != 0
    //          = |A - B| / A, B == 0, A != 0
    //            0            otherwise
    TECA_ALGORITHM_PROPERTY(double, tolerance)

protected:
    teca_dataset_diff();

    // Comparison methods.
    int compare_tables(const_p_teca_table table1, const_p_teca_table table2);

    int compare_cartesian_meshes(
        const_p_teca_cartesian_mesh reference_mesh,
        const_p_teca_cartesian_mesh data_mesh);

    int compare_array_collections(
        const_p_teca_array_collection reference_arrays,
        const_p_teca_array_collection data_arrays);

    int compare_arrays(
        const_p_teca_variant_array array1, const_p_teca_variant_array array2);

    // Reporting methods.

    // Call this with contextual information when datasets differ. You can use
    // printf formatting.
    void datasets_differ(const char* info, ...);

    // Push a frame onto our context stack.
    void push_frame(const std::string& frame);

    // Pop a frame off our context stack.
    void pop_frame();

private:
    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    // Tolerance for equality of field values.
    double tolerance;

    // Context stack for reporting.
    std::vector<std::string> stack;
};

#endif
