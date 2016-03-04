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

using std::vector;
using std::string;
using std::ostringstream;
using std::ofstream;
using std::cerr;
using std::endl;

// --------------------------------------------------------------------------
teca_dataset_diff::teca_dataset_diff()
    : tolerance(1e-12)
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
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for " + prefix + "(teca_dataset_diff)");
    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_dataset_diff::set_properties(const string &prefix, variables_map &opts)
{
}
#endif

// --------------------------------------------------------------------------
const_p_teca_dataset teca_dataset_diff::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
    (void) port;
    (void) request;

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
        datasets_differ("dataset 1 is empty, 2 is not.");
        return nullptr;
    }

    if (!input_data[0]->empty() && input_data[1]->empty())
    {
        datasets_differ("dataset 2 is empty, 1 is not.");
        return nullptr;
    }

    // If the datasets are both empty, they are "equal." :-/
    if (input_data[0]->empty() && input_data[1]->empty())
    {
        return nullptr;
    }

    // get the inputs. They can be tables or cartesian meshes.
    const_p_teca_table table1 = std::dynamic_pointer_cast<const teca_table>(input_data[0]);
    const_p_teca_table table2 = std::dynamic_pointer_cast<const teca_table>(input_data[1]);
    const_p_teca_cartesian_mesh mesh1 = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);
    const_p_teca_cartesian_mesh mesh2 = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[1]);

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
          return nullptr;
    }
    else 
    {
      if (this->compare_cartesian_meshes(mesh1, mesh1))
          return nullptr;
    }
    
    return nullptr;
}

// --------------------------------------------------------------------------
int teca_dataset_diff::compare_tables(
    const_p_teca_table table1,
    const_p_teca_table table2)
{
    // If the tables are different sizes, the datasets differ.
    if (table1->get_number_of_columns() != table2->get_number_of_columns())
    {
        datasets_differ("table 1 has %d columns while table 2 has %d columns.", 
                        table1->get_number_of_columns(), table2->get_number_of_columns());
        return 1;
    }
    if (table1->get_number_of_rows() != table2->get_number_of_rows())
    {
        datasets_differ("table 1 has %d rows while table 2 has %d rows.", 
                        table1->get_number_of_rows(), table2->get_number_of_rows());
        return 1;
    }

    // At this point, we know that the tables are both non-empty and the same size, 
    // so we simply compare them one element at a time.
    unsigned int ncols = table1->get_number_of_columns();
    for (unsigned int col = 0; col < ncols; ++col)
    {
        const_p_teca_variant_array col1 = table1->get_column(col);
        const_p_teca_variant_array col2 = table2->get_column(col);
        std::stringstream col_str;
        col_str << "Column " << col << " (" << table1->get_column_name(col) << ")" << std::ends;
        this->push_frame(col_str.str());
        int status = compare_arrays(col1, col2);
        this->pop_frame();
        if (status != 0)
          return status;
    }
 
    return 0;
}

// --------------------------------------------------------------------------
int teca_dataset_diff::compare_arrays(
    const_p_teca_variant_array array1, 
    const_p_teca_variant_array array2)
{
    // Arrays of different sizes are different.
    if (array1->size() != array2->size())
    {
        datasets_differ("arrays have different sizes.");
        return 1;
    }
   
    for (unsigned long i = 0; i < array1->size(); ++i)
    {
        TEMPLATE_DISPATCH_CLASS(const teca_variant_array_impl, double,
                                array1.get(), array2.get(), 
                                double v1 = p1_tt->get(i);
                                double v2 = p2_tt->get(i);
                                double abs_diff = std::abs(v1 - v2);
                                if (abs_diff > this->tolerance)
                                {
                                    datasets_differ("Absolute difference %g exceeds tolerance %g in element %d",
                                                    abs_diff, this->tolerance, i);
                                    return 1;
                                }
                                )
        TEMPLATE_DISPATCH_CLASS(const teca_variant_array_impl, float,
                                array1.get(), array2.get(), 
                                float v1 = p1_tt->get(i);
                                float v2 = p2_tt->get(i);
                                float abs_diff = std::abs(v1 - v2);
                                if (abs_diff > this->tolerance)
                                {
                                    datasets_differ("Absolute difference %g exceeds tolerance %g in element %d",
                                                    abs_diff, this->tolerance, i);
                                    return 1;
                                }
                                )
        TEMPLATE_DISPATCH_CLASS(const teca_variant_array_impl, long,
                                array1.get(), array2.get(), 
                                long v1 = p1_tt->get(i);
                                long v2 = p2_tt->get(i);
                                if (v1 != v2)
                                {
                                    datasets_differ("%d != %d in element %d",
                                                    v1, v2, i);
                                    return 1;
                                }
                                )
        TEMPLATE_DISPATCH_CLASS(const teca_variant_array_impl, int,
                                array1.get(), array2.get(), 
                                int v1 = p1_tt->get(i);
                                int v2 = p2_tt->get(i);
                                if (v1 != v2)
                                {
                                    datasets_differ("%d != %d in element %d",
                                                    v1, v2, i);
                                    return 1;
                                }
                                )
        TEMPLATE_DISPATCH_CLASS(const teca_variant_array_impl, string,
                                array1.get(), array2.get(),
                                string s1 = p1_tt->get(i);
                                string s2 = p2_tt->get(i);
                                if (s1 != s2)
                                {
                                     datasets_differ("'%s' != '%s' in element %d",
                                                     s1.c_str(), s2.c_str(), i);
                                     return 1;
                                }
                                )
    }
    return 0;
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
        datasets_differ("data array collection does not have %s, found in reference array collection.",
                        reference_arrays->get_name(i).c_str());
        return 1;
      }
    }
 
    // Now diff the contents.
    for (unsigned int i = 0; i < reference_arrays->size(); ++i)
    {
      const_p_teca_variant_array a1 = reference_arrays->get(i);
      string name = reference_arrays->get_name(i);
      const_p_teca_variant_array a2 = data_arrays->get(name);
      this->push_frame(name);
      int status = this->compare_arrays(a1, a2);
      this->pop_frame();
      if (status != 0)
        return status;
    }
    return 0;
}

// --------------------------------------------------------------------------
int teca_dataset_diff::compare_cartesian_meshes(
    const_p_teca_cartesian_mesh reference_mesh,
    const_p_teca_cartesian_mesh data_mesh)
{
    // If the meshes are different sizes, the datasets differ.
    if (reference_mesh->get_x_coordinates()->size() != data_mesh->get_x_coordinates()->size())
    {
      datasets_differ("data mesh has %d points in x, whereas reference mesh has %d.",
                      static_cast<int>(reference_mesh->get_x_coordinates()->size()), 
                      static_cast<int>(data_mesh->get_x_coordinates()->size()));
      return 1;
    }
    if (reference_mesh->get_y_coordinates()->size() != data_mesh->get_y_coordinates()->size())
    {
      datasets_differ("data mesh has %d points in y, whereas reference mesh has %d.",
                      static_cast<int>(reference_mesh->get_y_coordinates()->size()), 
                      static_cast<int>(data_mesh->get_y_coordinates()->size()));
      return 1;
    }
    if (reference_mesh->get_z_coordinates()->size() != data_mesh->get_z_coordinates()->size())
    {
      datasets_differ("data mesh has %d points in z, whereas reference mesh has %d.",
                      static_cast<int>(reference_mesh->get_z_coordinates()->size()), 
                      static_cast<int>(data_mesh->get_z_coordinates()->size()));
      return 1;
    }

    // If the arrays are different in shape or in content, the datasets differ.
    int status;
    const_p_teca_array_collection arrays1, arrays2;

    // Point arrays.
    arrays1 = reference_mesh->get_point_arrays();
    arrays2 = data_mesh->get_point_arrays();
    this->push_frame("Point arrays");
    status = this->compare_array_collections(arrays1, arrays2);
    this->pop_frame();
    if (status != 0)
      return status;
 
    // cell-centered arrays.
    arrays1 = reference_mesh->get_cell_arrays();
    arrays2 = data_mesh->get_cell_arrays();
    this->push_frame("Cell arrays");
    status = this->compare_array_collections(arrays1, arrays2);
    this->pop_frame();
    if (status != 0)
      return status;

    // Edge-centered arrays.
    arrays1 = reference_mesh->get_edge_arrays();
    arrays2 = data_mesh->get_edge_arrays();
    this->push_frame("Edge arrays");
    status = this->compare_array_collections(arrays1, arrays2);
    this->pop_frame();
    if (status != 0)
      return status;
 
    // Face-centered arrays.
    arrays1 = reference_mesh->get_face_arrays();
    arrays2 = data_mesh->get_face_arrays();
    this->push_frame("Face arrays");
    status = this->compare_array_collections(arrays1, arrays2);
    this->pop_frame();
    if (status != 0)
      return status;
 
    // Non-geometric arrays.
    arrays1 = reference_mesh->get_information_arrays();
    arrays2 = data_mesh->get_information_arrays();
    this->push_frame("Informational arrays");
    status = this->compare_array_collections(arrays1, arrays2);
    this->pop_frame();
    if (status != 0)
      return status;

    // Coordinate arrays.
    this->push_frame("X coordinates");
    status = this->compare_arrays(reference_mesh->get_x_coordinates(),
                                  data_mesh->get_x_coordinates());
    this->pop_frame();
    if (status != 0)
      return status;

    this->push_frame("Y coordinates");
    status = this->compare_arrays(reference_mesh->get_y_coordinates(),
                                  data_mesh->get_y_coordinates());
    this->pop_frame();
    if (status != 0)
      return status;

    this->push_frame("Z coordinates");
    status = this->compare_arrays(reference_mesh->get_z_coordinates(),
                                  data_mesh->get_z_coordinates());
    this->pop_frame();
    if (status != 0)
      return status;
 
    return 0;
}

void teca_dataset_diff::datasets_differ(const char* info, ...)
{
    // Assemble the informational message.
    char m[8192+1];
    va_list argp;
    va_start(argp, info);
    vsnprintf(m, 8192, info, argp);
    va_end(argp);

    // Send it to stderr with contextual information.
    for (size_t i = 0; i < this->stack.size(); ++i)
        std::cerr << this->stack[i] << "FAIL: ";
    std::cerr << m << std::endl;
}

void teca_dataset_diff::push_frame(const std::string& frame)
{
  this->stack.push_back(frame);
}

void teca_dataset_diff::pop_frame()
{
  this->stack.pop_back();
}

