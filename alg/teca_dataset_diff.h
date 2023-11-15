#ifndef teca_dataset_diff_h
#define teca_dataset_diff_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_table.h"
#include "teca_mesh.h"
#include "teca_cartesian_mesh.h"
#include "teca_curvilinear_mesh.h"
#include "teca_arakawa_c_grid.h"
#include "teca_array_collection.h"

#include <vector>
#include <string>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_dataset_diff)

/// compute the element wise difference between to datasets
/**
 * a two input algorithm that compares datasets by examining each element of their
 * contained arrays. a threshold is used to detect when an element is different. a
 * report containing the string FAIL is issued to stderr stream when a difference
 * is detected. this algorithm is the core of TECA's regression test suite.
 *
 * by convention the first input produces the reference dataset, and the second
 * input produces the dataset to validate. this is primarilly to support
 * map-reduce implementation where after the reduction only rank 0 has data.
*/
class TECA_EXPORT teca_dataset_diff : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_dataset_diff)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_dataset_diff)
    TECA_ALGORITHM_CLASS_NAME(teca_dataset_diff)
    ~teca_dataset_diff();

    // report/initialize to/from Boost program options objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name relative_tolerance
     * Relative tolerance below which two floating-point numbers a and b are
     * considered equal. if |a - b| <= max(|a|,|b|)*tol then a is equal to b.
     * the relative tolerance is used with numbers not close to zero.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, relative_tolerance)
    ///@}

    /** @name absolute_tolerance
     * The absolute tolerance below which two floating point numbers a and b
     * are considered equal. if |a - b| <= tol then a is equal to b. The
     * absolute tolerance is used with numbers close to zero.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, absolute_tolerance)
    ///@}

    /** @name skip_arrays
     * A list of arrays that are ignored during tests.
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, skip_array)
    ///@}

protected:
    teca_dataset_diff();

    // Comparison methods.
    int compare_tables(const_p_teca_table table1, const_p_teca_table table2);

    int compare_meshes(
        const_p_teca_mesh reference_mesh,
        const_p_teca_mesh data_mesh);

    int compare_cartesian_meshes(
        const_p_teca_cartesian_mesh reference_mesh,
        const_p_teca_cartesian_mesh data_mesh);

    int compare_curvilinear_meshes(
        const_p_teca_curvilinear_mesh reference_mesh,
        const_p_teca_curvilinear_mesh data_mesh);

    int compare_arakawa_c_grids(
        const_p_teca_arakawa_c_grid reference_mesh,
        const_p_teca_arakawa_c_grid data_mesh);

    int compare_array_collections(
        const_p_teca_array_collection reference_arrays,
        const_p_teca_array_collection data_arrays);

    // Reporting methods.

    // Call this with contextual information when datasets differ. You can use
    // printf formatting.
    void datasets_differ(const char* info, ...);

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

    double get_abs_tol() const;
    double get_rel_tol() const;

private:
    double relative_tolerance;
    double absolute_tolerance;
    std::vector<std::string> skip_arrays;
};

#endif
