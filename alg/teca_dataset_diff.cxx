#include "teca_dataset_diff.h"

#include "teca_table.h"
#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_metadata.h"
#include "teca_file_util.h"

#include <iostream>
#include <sstream>
#include <stdarg.h>
#include <cmath>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif


// --------------------------------------------------------------------------
teca_dataset_diff::teca_dataset_diff()
    : tolerance(1e-6)
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
        TECA_POPTS_GET(double, prefix, tolerance, "relative test tolerance")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_dataset_diff::set_properties(const std::string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, double, prefix, tolerance)
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_dataset_diff::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
    (void) port;

#if defined(TECA_HAS_MPI)
    int rank = 0;

    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
        MPI_Comm_rank(this->get_communicator(), &rank);
#endif

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

    // get input 1 initializer
    if (input_md[1].get("index_initializer_key", initializer_key))
    {
        TECA_ERROR("Input 0 metadata is missing index_initializer_key")
        return teca_metadata();
    }

    // if one were to run across all indices, both inputs would need to have
    // the same number of them. it is not necessarily an error to have
    // different numbers of indices because one could configure the executive
    // to run over a mutual subset

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

    // after map-reduce phase of a parallel run, only rank 0
    // will have data. we can assume that if the first input,
    // which by convention is the reference dataset, is empty
    // then the second one should be as well.
    if (!input_data[0] && !input_data[1])
        return nullptr;

    // We need exactly two non-NULL inputs to compute a difference.
    if (!input_data[0])
    {
        TECA_ERROR("Input dataset 1 is NULL.")
        return nullptr;
    }

    if (!input_data[1])
    {
        TECA_ERROR("Input dataset 2 is NULL.")
        return nullptr;
    }

    // If one dataset is empty but not the other, the datasets differ.
    if (input_data[0]->empty() && !input_data[1]->empty())
    {
        TECA_ERROR("dataset 1 is empty, 2 is not.")
        return nullptr;
    }

    if (!input_data[0]->empty() && input_data[1]->empty())
    {
        TECA_ERROR("dataset 2 is empty, 1 is not.")
        return nullptr;
    }

    // If the datasets are both empty, they are "equal." :-/
    if (input_data[0]->empty() && input_data[1]->empty())
        return nullptr;

    // get the inputs. They can be tables or cartesian meshes.
    const_p_teca_table table1 =
        std::dynamic_pointer_cast<const teca_table>(input_data[0]);

    const_p_teca_table table2 =
         std::dynamic_pointer_cast<const teca_table>(input_data[1]);

    const_p_teca_cartesian_mesh mesh1 =
        std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);

    const_p_teca_cartesian_mesh mesh2 =
        std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[1]);

    // No mixed types!
    if (((table1 && !table2) || (!table1 && table2)) ||
        ((mesh1 && !mesh2) || (!mesh1 && mesh2)))
    {
        TECA_ERROR("input datasets must have matching types.");
        return nullptr;
    }

    if (!table1 && !mesh1)
    {
        TECA_ERROR("input datasets must be teca_tables or teca_cartesian_meshes.")
        return nullptr;
    }

    if (table1)
    {
        if (this->compare_tables(table1, table2))
        {
            TECA_ERROR("Failed to compare tables.");
            return nullptr;
        }
    }
    else
    {
        if (this->compare_cartesian_meshes(mesh1, mesh2))
        {
            TECA_ERROR("Failed to compare cartesian meshes.");
            return nullptr;
        }
    }

    return nullptr;
}

// --------------------------------------------------------------------------
int teca_dataset_diff::compare_tables(
    const_p_teca_table table1,
    const_p_teca_table table2)
{
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
    for (unsigned int col = 0; col < ncols1; ++col)
    {
        const_p_teca_variant_array col1 = table1->get_column(col);
        const_p_teca_variant_array col2 = table2->get_column(col);
        if (compare_arrays(col1, col2))
        {
            TECA_ERROR("difference in column " << col << " \""
                << table1->get_column_name(col) << "\"")
            return -1;

        }
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_dataset_diff::compare_arrays(
    const_p_teca_variant_array array1,
    const_p_teca_variant_array array2)
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

        for (size_t i = 0; i < n_elem; ++i)
        {
            // we don't care too much about performance here so
            // use double precision for the comparison.
            double ref_val = static_cast<double>(pa1[i]);  // reference
            double comp_val = static_cast<double>(pa2[i]); // computed

            // Compute the relative difference.
            double rel_diff = 0.0;
            if (ref_val != 0.0)
                rel_diff = std::abs(comp_val - ref_val) / std::abs(ref_val);
            else if (comp_val != 0.0)
                rel_diff = std::abs(comp_val - ref_val) / std::abs(comp_val);

            if (rel_diff > this->tolerance)
            {
                TECA_ERROR("relative difference " << rel_diff << " exceeds tolerance "
                    << this->tolerance << " in element " << i << ". ref value \""
                    << ref_val << "\" is not equal to test value \"" << comp_val << "\"")
                return -1;
            }
        }

        // we are here, arrays are the same
        return 0;
        )
    // handle arrays of strings
    TEMPLATE_DISPATCH_CLASS(
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
                    TECA_ERROR("string element " << i << " not equal. ref value \"" << v1
                        << "\" is not equal to test value \"" << v2 << "\"")
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
    for (unsigned int i = 0; i < reference_arrays->size(); ++i)
    {
        const_p_teca_variant_array a1 = reference_arrays->get(i);
        std::string name = reference_arrays->get_name(i);
        const_p_teca_variant_array a2 = data_arrays->get(name);
        if (this->compare_arrays(a1, a2))
        {
            TECA_ERROR("difference in array " << i << " \"" << name << "\"")
            return -1;
        }
    }
    return 0;
}

// --------------------------------------------------------------------------
int teca_dataset_diff::compare_cartesian_meshes(
    const_p_teca_cartesian_mesh reference_mesh,
    const_p_teca_cartesian_mesh data_mesh)
{
    // If the meshes are different sizes, the datasets differ.
    if (reference_mesh->get_x_coordinates()->size()
        != data_mesh->get_x_coordinates()->size())
    {
        TECA_ERROR("data mesh has " << data_mesh->get_x_coordinates()->size()
            << " points in x, whereas reference mesh has "
            << reference_mesh->get_x_coordinates()->size() << ".")
        return -1;
    }
    if (reference_mesh->get_y_coordinates()->size()
        != data_mesh->get_y_coordinates()->size())
    {
        TECA_ERROR("data mesh has " << data_mesh->get_y_coordinates()->size()
            << " points in y, whereas reference mesh has "
            << reference_mesh->get_y_coordinates()->size() << ".")
        return -1;
    }
    if (reference_mesh->get_z_coordinates()->size()
        != data_mesh->get_z_coordinates()->size())
    {
        TECA_ERROR("data mesh has " << data_mesh->get_z_coordinates()->size()
            << " points in z, whereas reference mesh has "
            << reference_mesh->get_z_coordinates()->size() << ".")
        return -1;
    }

    // If the arrays are different in shape or in content, the datasets differ.
    const_p_teca_array_collection arrays1, arrays2;

    // Point arrays.
    arrays1 = reference_mesh->get_point_arrays();
    arrays2 = data_mesh->get_point_arrays();
    if (this->compare_array_collections(arrays1, arrays2))
    {
        TECA_ERROR("difference in point arrays")
        return -1;
    }

    // cell-centered arrays.
    arrays1 = reference_mesh->get_cell_arrays();
    arrays2 = data_mesh->get_cell_arrays();
    if (this->compare_array_collections(arrays1, arrays2))
    {
        TECA_ERROR("difference in cell arrays")
        return -1;
    }

    // Edge-centered arrays.
    arrays1 = reference_mesh->get_edge_arrays();
    arrays2 = data_mesh->get_edge_arrays();
    if (this->compare_array_collections(arrays1, arrays2))
    {
        TECA_ERROR("difference in edge arrays")
        return -1;
    }

    // Face-centered arrays.
    arrays1 = reference_mesh->get_face_arrays();
    arrays2 = data_mesh->get_face_arrays();
    if (this->compare_array_collections(arrays1, arrays2))
    {
        TECA_ERROR("difference in face arrays")
        return -1;
    }

    // Non-geometric arrays.
    arrays1 = reference_mesh->get_information_arrays();
    arrays2 = data_mesh->get_information_arrays();
    if (this->compare_array_collections(arrays1, arrays2))
    {
        TECA_ERROR("differrnce in informational arrays")
        return -1;
    }

    // Coordinate arrays.
    if (this->compare_arrays(reference_mesh->get_x_coordinates(),
        data_mesh->get_x_coordinates()))
    {
        TECA_ERROR("difference in x coordinates")
        return -1;
    }

    if (this->compare_arrays(reference_mesh->get_y_coordinates(),
        data_mesh->get_y_coordinates()))
    {
        TECA_ERROR("difference in y coordinates")
        return -1;
    }

    if (this->compare_arrays(reference_mesh->get_z_coordinates(),
        data_mesh->get_z_coordinates()))
    {
        TECA_ERROR("difference in z coordinates")
        return -1;
    }

    return 0;
}
