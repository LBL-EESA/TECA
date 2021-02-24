#include "teca_dataset_diff.h"

#include "teca_table.h"
#include "teca_cartesian_mesh.h"
#include "teca_curvilinear_mesh.h"
#include "teca_arakawa_c_grid.h"
#include "teca_array_collection.h"
#include "teca_metadata.h"
#include "teca_file_util.h"
#include "teca_coordinate_util.h"
#include "teca_mpi.h"

#include <iostream>
#include <sstream>
#include <stdarg.h>
#include <cmath>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <iostream>
#include <iomanip>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#define TEST_STATUS(_msg)                               \
    std::cerr << teca_parallel_id()                     \
        << " teca_dataset_diff :: " _msg << std::endl;

// --------------------------------------------------------------------------
teca_dataset_diff::teca_dataset_diff()
    : relative_tolerance(1.0e-6), absolute_tolerance(-1.0)
{
    this->set_number_of_input_connections(2);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_dataset_diff::~teca_dataset_diff()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_dataset_diff::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_dataset_diff":prefix));

    opts.add_options()
        TECA_POPTS_GET(double, prefix, relative_tolerance, "relative test tolerance")
        TECA_POPTS_GET(double, prefix, absolute_tolerance, "absolute test tolerance")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_dataset_diff::set_properties(const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, double, prefix, relative_tolerance)
    TECA_POPTS_SET(opts, double, prefix, absolute_tolerance)
    TECA_POPTS_SET(opts, int, prefix, verbose)
}
#endif

// --------------------------------------------------------------------------
double teca_dataset_diff::get_abs_tol() const
{
    return this->absolute_tolerance <= 0.0 ?
        teca_coordinate_util::equal_tt<double>::absTol() :
        this->absolute_tolerance;
}

// --------------------------------------------------------------------------
double teca_dataset_diff::get_rel_tol() const
{
    return this->relative_tolerance <= 0.0 ?
        teca_coordinate_util::equal_tt<double>::relTol() :
        this->relative_tolerance;
}

// --------------------------------------------------------------------------
teca_metadata teca_dataset_diff::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
    (void) port;

    // get input 0 initializer
    std::string initializer_key;
    if (input_md[0].get("index_initializer_key", initializer_key))
    {
        TECA_ERROR("Input 0 metadata is missing index_initializer_key")
        return teca_metadata();
    }

    unsigned long n_indices_0 = 0;
    if (input_md[0].get(initializer_key, n_indices_0))
    {
        TECA_ERROR("Input 0 metadata is missing its intializer \""
            << initializer_key << "\"")
        return teca_metadata();
    }

    // if one were to run across all indices, both inputs would need to have
    // the same number of them. it is not necessarily an error to have
    // different numbers of indices because one could configure the executive
    // to run over a mutual subset
    /*
    // get input 1 initializer
    if (input_md[1].get("index_initializer_key", initializer_key))
    {
        TECA_ERROR("Input 1 metadata is missing index_initializer_key")
        return teca_metadata();
    }

    unsigned long n_indices_1 = 0;
    if (input_md[1].get(initializer_key, n_indices_1))
    {
        TECA_ERROR("Input 0 metadata is missing its intializer \""
            << initializer_key << "\"")
        return teca_metadata();
    }
    */

    // prepare pieline executive metadata to run a test for each input dataset
    teca_metadata omd(input_md[0]);
    omd.set("index_initializer_key", std::string("number_of_tests"));
    omd.set("index_request_key", std::string("test_id"));
    omd.set("number_of_tests", n_indices_0);

    return omd;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_dataset_diff::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void) port;

    std::vector<teca_metadata> up_reqs;

    // get the current index
    unsigned long test_id = 0;
    if (request.get("test_id", test_id))
    {
        TECA_ERROR("Request is missing the index_request_key test_id")
        return up_reqs;
    }

    // get input 0 request key
    std::string request_key;
    if (input_md[0].get("index_request_key", request_key))
    {
        TECA_ERROR("Input 0 metadata is missing index_request_key")
        return up_reqs;
    }

    // make the request for input 0
    teca_metadata req_0(request);
    req_0.set("index_request_key", request_key);
    req_0.set(request_key, test_id);
    req_0.remove("test_id");

    // get input 1 request key
    if (input_md[1].get("index_request_key", request_key))
    {
        TECA_ERROR("Input 1 metadata is missing index_request_key")
        return up_reqs;
    }

    // make the request for input 1
    teca_metadata req_1(request);
    req_1.set("index_request_key", request_key);
    req_1.set(request_key, test_id);
    req_1.remove("test_id");

    // send them upstream
    up_reqs.push_back(req_0);
    up_reqs.push_back(req_1);
    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_dataset_diff::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
    (void) port;
    (void) request;

    int rank = 0;
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
        MPI_Comm_rank(this->get_communicator(), &rank);
#endif

    const_p_teca_dataset ds0 = input_data[0];
    const_p_teca_dataset ds1 = input_data[1];

    // after map-reduce phase of a parallel run, only rank 0
    // will have data. we can assume that if the first input,
    // which by convention is the reference dataset, is empty
    // then the second one should be as well.
    if (!ds0 && !ds1)
        return nullptr;

    // We need exactly two non-NULL inputs to compute a difference.
    if (!ds0)
    {
        TECA_ERROR("Input dataset 1 is NULL.")
        return nullptr;
    }

    if (!ds1)
    {
        TECA_ERROR("Input dataset 2 is NULL.")
        return nullptr;
    }

    // If one dataset is empty but not the other, the datasets differ.
    if (ds0->empty() && !ds1->empty())
    {
        TECA_ERROR("dataset 1 is empty, 2 is not.")
        return nullptr;
    }

    if (!ds0->empty() && ds1->empty())
    {
        TECA_ERROR("dataset 2 is empty, 1 is not.")
        return nullptr;
    }

    // If the datasets are both empty, they are "equal." :-/
    if (ds0->empty() && ds1->empty())
    {
        if (rank == 0)
        {
            TECA_ERROR("Both the reference and test datasets are empty")
        }
        return nullptr;
    }

    // compare the inputs. the type of data is inferred from the
    // reference mesh.
    if (dynamic_cast<const teca_table*>(ds0.get()))
    {
        if (this->compare_tables(
            std::dynamic_pointer_cast<const teca_table>(ds0),
            std::dynamic_pointer_cast<const teca_table>(ds1)))
        {
            TECA_ERROR("Failed to compare tables.");
            return nullptr;
        }
    }
    else if (dynamic_cast<const teca_cartesian_mesh*>(ds0.get()))
    {
        if (this->compare_cartesian_meshes(
            std::dynamic_pointer_cast<const teca_cartesian_mesh>(ds0),
            std::dynamic_pointer_cast<const teca_cartesian_mesh>(ds1)))
        {
            TECA_ERROR("Failed to compare cartesian_meshes.");
            return nullptr;
        }
    }
    else if (dynamic_cast<const teca_curvilinear_mesh*>(ds0.get()))
    {
        if (this->compare_curvilinear_meshes(
            std::dynamic_pointer_cast<const teca_curvilinear_mesh>(ds0),
            std::dynamic_pointer_cast<const teca_curvilinear_mesh>(ds1)))
        {
            TECA_ERROR("Failed to compare curvilinear_meshes.");
            return nullptr;
        }
    }
    else if (dynamic_cast<const teca_arakawa_c_grid*>(ds0.get()))
    {
        if (this->compare_arakawa_c_grids(
            std::dynamic_pointer_cast<const teca_arakawa_c_grid>(ds0),
            std::dynamic_pointer_cast<const teca_arakawa_c_grid>(ds1)))
        {
            TECA_ERROR("Failed to compare arakawa_c_grids.");
            return nullptr;
        }
    }
    else
    {
        TECA_ERROR("Unsupported dataset type \""
            << ds0->get_class_name() << "\"")
        return nullptr;
    }

    return nullptr;
}

// --------------------------------------------------------------------------
int teca_dataset_diff::compare_tables(
    const_p_teca_table table1,
    const_p_teca_table table2)
{
    int rank = 0;
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
        MPI_Comm_rank(this->get_communicator(), &rank);
#endif
    if (this->verbose && (rank == 0))
    {
        TEST_STATUS("comparing tables")
    }

    unsigned int ncols1 = table1->get_number_of_columns();
    unsigned int ncols2 = table2->get_number_of_columns();

    // If the tables are different sizes, the datasets differ.
    if (ncols1 != ncols2)
    {
        const_p_teca_table bigger = ncols1 > ncols2 ? table1 : table2;
        const_p_teca_table smaller = ncols1 <= ncols2 ? table1 : table2;
        unsigned int ncols = ncols1 > ncols2 ? ncols1 : ncols2;

        std::ostringstream oss;
        for (unsigned int i = 0; i < ncols; ++i)
        {
            std::string colname = bigger->get_column_name(i);
            if (!smaller->has_column(colname))
                oss << (oss.tellp()?", \"":"\"") << colname << "\"";
        }

        TECA_ERROR("The baseline table has " << ncols1
            << " columns while test table has " << ncols2
            << " columns. Columns " << oss.str() << " are missing")
        return -1;
    }

    if (table1->get_number_of_rows() != table2->get_number_of_rows())
    {
        TECA_ERROR("The baseline table has " << table1->get_number_of_rows()
            << " rows while test table has " << table2->get_number_of_rows()
            << " rows.")
        return -1;
    }

    // At this point, we know that the tables are both non-empty and the same size,
    // so we simply compare them one element at a time.
    double absTol = this->get_abs_tol();
    double relTol = this->get_rel_tol();

    for (unsigned int col = 0; col < ncols1; ++col)
    {
        const_p_teca_variant_array col1 = table1->get_column(col);
        const_p_teca_variant_array col2 = table2->get_column(col);

        const std::string &col_name = table1->get_column_name(col);

        if (this->verbose && (rank == 0))
        {
            TEST_STATUS("  comparing collumn \"" << col_name
                << "\" absTol=" << max_prec(double) << absTol
                << " relTol=" << max_prec(double) << relTol)
        }

        if (compare_arrays(col1, col2, absTol, relTol))
        {

            TECA_ERROR("difference in column " << col << " \"" << col_name << "\"")
            return -1;
        }
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_dataset_diff::compare_arrays(
    const_p_teca_variant_array array1,
    const_p_teca_variant_array array2,
    double absTol, double relTol)
{
    // Arrays of different sizes are different.
    size_t n_elem = array1->size();
    if (n_elem != array2->size())
    {
        TECA_ERROR("arrays have different sizes "
            << n_elem << " and " << array2->size())
        return -1;
    }

    // handle POD arrays
    TEMPLATE_DISPATCH(const teca_variant_array_impl,
        array1.get(),

        // we know the type of array 1 now,
        // check the type of array 2
        const TT *a2 = dynamic_cast<const TT*>(array2.get());
        if (!a2)
        {
            TECA_ERROR("arrays have different element types.")
            return -1;
        }

        // compare elements
        const NT *pa1 = static_cast<const TT*>(array1.get())->get();
        const NT *pa2 = a2->get();

        std::string diagnostic;
        for (size_t i = 0; i < n_elem; ++i)
        {
            if (std::isinf(pa1[i]) && std::isinf(pa2[i]))
            {
                // the GFDL TC tracker returns inf for some fields in some cases.
                // warn about it so that it may be addressed in other algorithms.
                if (this->verbose)
                {
                    TECA_WARNING("Inf detected in element " << i)
                }
            }
            else if (std::isnan(pa1[i]) || std::isnan(pa2[i]))
            {
                // for the time being, don't allow NaN.
                TECA_ERROR("NaN detected in element " << i)
                return -1;
            }
            else if (!teca_coordinate_util::equal<double>(pa1[i], pa2[i],
                diagnostic, relTol, absTol))
            {
                TECA_ERROR("difference above the prescribed tolerance detected"
                    " in element " << i << ". " << diagnostic)
                return -1;
            }
        }

        // we are here, arrays are the same
        return 0;
        )
    // handle arrays of strings
    TEMPLATE_DISPATCH_CASE(
        const teca_variant_array_impl, std::string,
        array1.get(),
        if (dynamic_cast<const TT*>(array2.get()))
        {
            const TT *a1 = static_cast<const TT*>(array1.get());
            const TT *a2 = static_cast<const TT*>(array2.get());

            for (size_t i = 0; i < n_elem; ++i)
            {
                // compare elements
                const std::string &v1 = a1->get(i);
                const std::string &v2 = a2->get(i);
                if (v1 != v2)
                {
                    TECA_ERROR("string element " << i << " not equal. ref value \""
                        << v1 << "\" is not equal to test value \"" << v2 << "\"")
                    return -1;
                }
            }

            // we are here, arrays are the same
            return 0;
        }
        )

    // we are here, array 1 type is not handled
    TECA_ERROR("diff for the element type of "
        "array1 is not implemented.")
    return -1;
}

// --------------------------------------------------------------------------
int teca_dataset_diff::compare_array_collections(
    const_p_teca_array_collection reference_arrays,
    const_p_teca_array_collection data_arrays)
{
    int rank = 0;
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
        MPI_Comm_rank(this->get_communicator(), &rank);
#endif

    // The data arrays should contain all the data in the reference arrays.
    for (unsigned int i = 0; i < reference_arrays->size(); ++i)
    {
        if (!data_arrays->has(reference_arrays->get_name(i)))
        {
            TECA_ERROR("data array collection does not have array \""
                 << reference_arrays->get_name(i)
                 << "\" from the reference array collection.")
            return -1;
         }
    }

    // Now diff the contents.
    double absTol = this->get_abs_tol();
    double relTol = this->get_rel_tol();

    for (unsigned int i = 0; i < reference_arrays->size(); ++i)
    {
        const_p_teca_variant_array a1 = reference_arrays->get(i);
        std::string name = reference_arrays->get_name(i);

        const_p_teca_variant_array a2 = data_arrays->get(name);

        if (this->verbose && (rank == 0))
        {
            TEST_STATUS("    comparing array \"" << name
                << "\" absTol=" << max_prec(double) << absTol
                << " relTol=" << max_prec(double) << relTol)
        }

        if (this->compare_arrays(a1, a2, absTol, relTol))
        {
            TECA_ERROR("difference in array " << i << " \"" << name << "\"")
            return -1;
        }
    }
    return 0;
}

// --------------------------------------------------------------------------
int teca_dataset_diff::compare_meshes(
    const_p_teca_mesh reference_mesh,
    const_p_teca_mesh data_mesh)
{
    int rank = 0;
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
        MPI_Comm_rank(this->get_communicator(), &rank);
#endif

    // If the arrays are different in shape or in content, the datasets differ.
    const_p_teca_array_collection arrays1, arrays2;

    // Point arrays.
    arrays1 = reference_mesh->get_point_arrays();
    arrays2 = data_mesh->get_point_arrays();
    if (this->verbose && (rank == 0) && arrays1->size())
    {
        TEST_STATUS("  comparing point arrays")
    }
    if (this->compare_array_collections(arrays1, arrays2))
    {
        TECA_ERROR("difference in point arrays")
        return -1;
    }

    // cell-centered arrays.
    arrays1 = reference_mesh->get_cell_arrays();
    arrays2 = data_mesh->get_cell_arrays();
    if (this->verbose && (rank == 0) && arrays1->size())
    {
        TEST_STATUS("  comparing cell arrays")
    }
    if (this->compare_array_collections(arrays1, arrays2))
    {
        TECA_ERROR("difference in cell arrays")
        return -1;
    }

    // Edge-centered arrays.
    arrays1 = reference_mesh->get_x_edge_arrays();
    arrays2 = data_mesh->get_x_edge_arrays();
    if (this->verbose && (rank == 0) && arrays1->size())
    {
        TEST_STATUS("  comparing x-dege arrays")
    }
    if (this->compare_array_collections(arrays1, arrays2))
    {
        TECA_ERROR("difference in x-edge arrays")
        return -1;
    }

    arrays1 = reference_mesh->get_y_edge_arrays();
    arrays2 = data_mesh->get_y_edge_arrays();
    if (this->verbose && (rank == 0) && arrays1->size())
    {
        TEST_STATUS("  comparing y-edge arrays")
    }
    if (this->compare_array_collections(arrays1, arrays2))
    {
        TECA_ERROR("difference in y-edge arrays")
        return -1;
    }

    arrays1 = reference_mesh->get_z_edge_arrays();
    arrays2 = data_mesh->get_z_edge_arrays();
    if (this->verbose && (rank == 0) && arrays1->size())
    {
        TEST_STATUS("  comparing z-edge arrays")
    }
    if (this->compare_array_collections(arrays1, arrays2))
    {
        TECA_ERROR("difference in z-edge arrays")
        return -1;
    }

    // Face-centered arrays.
    arrays1 = reference_mesh->get_x_face_arrays();
    arrays2 = data_mesh->get_x_face_arrays();
    if (this->verbose && (rank == 0) && arrays1->size())
    {
        TEST_STATUS("  comparing x-face arrays")
    }
    if (this->compare_array_collections(arrays1, arrays2))
    {
        TECA_ERROR("difference in x-face arrays")
        return -1;
    }

    arrays1 = reference_mesh->get_y_face_arrays();
    arrays2 = data_mesh->get_y_face_arrays();
    if (this->verbose && (rank == 0) && arrays1->size())
    {
        TEST_STATUS("  comparing y-face arrays")
    }
    if (this->compare_array_collections(arrays1, arrays2))
    {
        TECA_ERROR("difference in y-face arrays")
        return -1;
    }

    arrays1 = reference_mesh->get_z_face_arrays();
    arrays2 = data_mesh->get_z_face_arrays();
    if (this->verbose && (rank == 0) && arrays1->size())
    {
        TEST_STATUS("  comparing z-face arrays")
    }
    if (this->compare_array_collections(arrays1, arrays2))
    {
        TECA_ERROR("difference in z-face arrays")
        return -1;
    }

    // Non-geometric arrays.
    arrays1 = reference_mesh->get_information_arrays();
    arrays2 = data_mesh->get_information_arrays();
    if (this->verbose && (rank == 0) && arrays1->size())
    {
        TEST_STATUS("  comparing information arrays")
    }
    if (this->compare_array_collections(arrays1, arrays2))
    {
        TECA_ERROR("difference in information arrays")
        return -1;
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_dataset_diff::compare_cartesian_meshes(
    const_p_teca_cartesian_mesh reference_mesh,
    const_p_teca_cartesian_mesh data_mesh)
{
    int rank = 0;
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
        MPI_Comm_rank(this->get_communicator(), &rank);
#endif

    // compare base class elements
    if (this->verbose && (rank == 0))
    {
        TEST_STATUS("comparing cartesian meshes")
    }
    if (this->compare_meshes(reference_mesh, data_mesh))
    {
        TECA_ERROR("Difference in mesh")
        return -1;
    }

    // Coordinate arrays.
    double absTol = this->get_abs_tol();
    double relTol = this->get_rel_tol();

    std::string name;
    const_p_teca_variant_array coord1 = reference_mesh->get_x_coordinates();
    reference_mesh->get_x_coordinate_variable(name);
    if (this->verbose && (rank == 0) && coord1->size())
    {
        TEST_STATUS("comparing x-coordinates " << name
            << " absTol=" << max_prec(double) << absTol
            << " relTol=" << max_prec(double) << relTol)
    }
    if (this->compare_arrays(coord1, data_mesh->get_x_coordinates(), absTol, relTol))
    {
        TECA_ERROR("difference in x coordinates")
        return -1;
    }

    coord1 = reference_mesh->get_y_coordinates();
    reference_mesh->get_y_coordinate_variable(name);
    if (this->verbose && (rank == 0) && coord1->size())
    {
        TEST_STATUS("comparing y-coordinates " << name
            << " absTol=" << max_prec(double) << absTol
            << " relTol=" << max_prec(double) << relTol)
    }
    if (this->compare_arrays(coord1, data_mesh->get_y_coordinates(), absTol, relTol))
    {
        TECA_ERROR("difference in y coordinates")
        return -1;
    }

    coord1 = reference_mesh->get_z_coordinates();
    reference_mesh->get_z_coordinate_variable(name);
    if (this->verbose && (rank == 0) && coord1->size())
    {
        TEST_STATUS("comparing z-coordinates " << name
            << " absTol=" << max_prec(double) << absTol
            << " relTol=" << max_prec(double) << relTol)
    }
    if (this->compare_arrays(coord1,
        data_mesh->get_z_coordinates(), absTol, relTol))
    {
        TECA_ERROR("difference in z coordinates")
        return -1;
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_dataset_diff::compare_curvilinear_meshes(
    const_p_teca_curvilinear_mesh reference_mesh,
    const_p_teca_curvilinear_mesh data_mesh)
{
    int rank = 0;
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
        MPI_Comm_rank(this->get_communicator(), &rank);
#endif

    // compare base class elements
    if (this->verbose && (rank == 0))
    {
        TEST_STATUS("comparing curvilinear meshes")
    }
    if (this->compare_meshes(reference_mesh, data_mesh))
    {
        TECA_ERROR("Difference in mesh")
        return -1;
    }

    // Coordinate arrays.
    double absTol = this->get_abs_tol();
    double relTol = this->get_rel_tol();

    if (this->verbose && (rank == 0))
    {
        TEST_STATUS("comparing x-coordinates"
            << " absTol=" << max_prec(double) << absTol
            << " relTol=" << max_prec(double) << relTol)
    }
    if (this->compare_arrays(reference_mesh->get_x_coordinates(),
        data_mesh->get_x_coordinates(), absTol, relTol))
    {
        TECA_ERROR("difference in x coordinates")
        return -1;
    }

    if (this->verbose && (rank == 0))
    {
        TEST_STATUS("comparing y-coordinates"
            << " absTol=" << max_prec(double) << absTol
            << " relTol=" << max_prec(double) << relTol)
    }
    if (this->compare_arrays(reference_mesh->get_y_coordinates(),
        data_mesh->get_y_coordinates(), absTol, relTol))
    {
        TECA_ERROR("difference in y coordinates")
        return -1;
    }

    if (this->verbose && (rank == 0))
    {
        TEST_STATUS("comparing z-coordinates"
            << " absTol=" << max_prec(double) << absTol
            << " relTol=" << max_prec(double) << relTol)
    }
    if (this->compare_arrays(reference_mesh->get_z_coordinates(),
        data_mesh->get_z_coordinates(), absTol, relTol))
    {
        TECA_ERROR("difference in z coordinates")
        return -1;
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_dataset_diff::compare_arakawa_c_grids(
    const_p_teca_arakawa_c_grid reference_mesh,
    const_p_teca_arakawa_c_grid data_mesh)
{
    int rank = 0;
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
        MPI_Comm_rank(this->get_communicator(), &rank);
#endif

    // compare base class elements
    if (this->verbose && (rank == 0))
    {
        TEST_STATUS("comparing arakawa c grids")
    }
    if (this->compare_meshes(reference_mesh, data_mesh))
    {
        TECA_ERROR("Difference in mesh")
        return -1;
    }

    // Coordinate arrays.
    double absTol = this->get_abs_tol();
    double relTol = this->get_rel_tol();

    if (this->verbose && (rank == 0))
    {
        TEST_STATUS("comparing m x-coordinates"
            << " absTol=" << max_prec(double) << absTol
            << " relTol=" << max_prec(double) << relTol)
    }
    if (this->compare_arrays(reference_mesh->get_m_x_coordinates(),
        data_mesh->get_m_x_coordinates(), absTol, relTol))
    {
        TECA_ERROR("difference in m_x coordinates")
        return -1;
    }

    if (this->verbose && (rank == 0))
    {
        TEST_STATUS("comparing m y-coordinates"
            << " absTol=" << max_prec(double) << absTol
            << " relTol=" << max_prec(double) << relTol)
    }
    if (this->compare_arrays(reference_mesh->get_m_y_coordinates(),
        data_mesh->get_m_y_coordinates(), absTol, relTol))
    {
        TECA_ERROR("difference in m_y coordinates")
        return -1;
    }

    if (this->verbose && (rank == 0))
    {
        TEST_STATUS("comparing u x-coordinates"
            << " absTol=" << max_prec(double) << absTol
            << " relTol=" << max_prec(double) << relTol)
    }
    if (this->compare_arrays(reference_mesh->get_u_x_coordinates(),
        data_mesh->get_u_x_coordinates(), absTol, relTol))
    {
        TECA_ERROR("difference in u_x coordinates")
        return -1;
    }

    if (this->verbose && (rank == 0))
    {
        TEST_STATUS("comparing u x-coordinates"
            << " absTol=" << max_prec(double) << absTol
            << " relTol=" << max_prec(double) << relTol)
    }
    if (this->compare_arrays(reference_mesh->get_u_y_coordinates(),
        data_mesh->get_u_y_coordinates(), absTol, relTol))
    {
        TECA_ERROR("difference in u_y coordinates")
        return -1;
    }

    if (this->verbose && (rank == 0))
    {
        TEST_STATUS("comparing v x-coordinates"
            << " absTol=" << max_prec(double) << absTol
            << " relTol=" << max_prec(double) << relTol)
    }
    if (this->compare_arrays(reference_mesh->get_v_x_coordinates(),
        data_mesh->get_v_x_coordinates(), absTol, relTol))
    {
        TECA_ERROR("difference in v_x coordinates")
        return -1;
    }

    if (this->verbose && (rank == 0))
    {
        TEST_STATUS("comparing v y-coordinates"
            << " absTol=" << max_prec(double) << absTol
            << " relTol=" << max_prec(double) << relTol)
    }
    if (this->compare_arrays(reference_mesh->get_v_y_coordinates(),
        data_mesh->get_v_y_coordinates(), absTol, relTol))
    {
        TECA_ERROR("difference in v_y coordinates")
        return -1;
    }

    if (this->verbose && (rank == 0))
    {
        TEST_STATUS("comparing m z-coordinates"
            << " absTol=" << max_prec(double) << absTol
            << " relTol=" << max_prec(double) << relTol)
    }
    if (this->compare_arrays(reference_mesh->get_m_z_coordinates(),
        data_mesh->get_m_z_coordinates(), absTol, relTol))
    {
        TECA_ERROR("difference in m_z coordinates")
        return -1;
    }

    if (this->verbose && (rank == 0))
    {
        TEST_STATUS("comparing w z-coordinates"
            << " absTol=" << max_prec(double) << absTol
            << " relTol=" << max_prec(double) << relTol)
    }
    if (this->compare_arrays(reference_mesh->get_w_z_coordinates(),
        data_mesh->get_w_z_coordinates(), absTol, relTol))
    {
        TECA_ERROR("difference in w_z coordinates")
        return -1;
    }

    return 0;
}
