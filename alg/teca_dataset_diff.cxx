#include "teca_dataset_diff.h"

#include "teca_table.h"
#include "teca_cartesian_mesh.h"
#include "teca_metadata.h"
#include "teca_file_util.h"

#include <iostream>
#include <sstream>
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

    opts.add_options()
        TECA_POPTS_GET(string, prefix, file_name, "path/name of file to write")
        TECA_POPTS_GET(bool, prefix, binary_mode, "write binary")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_dataset_diff::set_properties(const string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, string, prefix, file_name)
    TECA_POPTS_SET(opts, bool, prefix, binary_mode)
}
#endif

// --------------------------------------------------------------------------
const_p_teca_dataset teca_dataset_diff::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
    (void) port;

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
        std::cout << "Datasets differ (1 is empty, 2 is not)." << std::endl;
        return nullptr;
    }

    if (!input_data[0]->empty() && input_data[1]->empty())
    {
        std::cout << "Datasets differ (2 is empty, 1 is not)." << std::endl;
        return nullptr;
    }

    // If the datasets are both empty, they are "equal." :-/
    if (input_data[0]->empty() && input_data[1]->empty())
    {
        std::cout << "Datasets are equal." << std::endl;
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
      {
          TECA_ERROR("Failed to compare tables.");
          return nullptr;
      }
    }
    else 
    {
      if (this->compare_cartesian_meshes(mesh1, mesh1))
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
    // If the tables are different sizes, the datasets differ.
    if (table1->get_number_of_columns() != table2->get_number_of_columns())
    {
        std::cout << "Datasets differ (table 1 has " 
                  << table1->get_number_of_columns() 
                  << " columns while table 2 has "
                  << table2->get_number_of_columns()
                  << ")." << std::endl;
        return 0;
    }
    if (table1->get_number_of_rows() != table2->get_number_of_rows())
    {
        std::cout << "Datasets differ (table 1 has " 
                  << table1->get_number_of_rows() 
                  << " rows while table 2 has "
                  << table2->get_number_of_rows()
                  << ")." << std::endl;
        return 0;
    }

    // At this point, we know that the tables are both non-empty and the same size, 
    // so we simply compare them one element at a time.
    // FIXME: For now, we use the max norm and compare it to our tolerance.
    double max_abs = 0.0;
    unsigned int max_abs_diff_col = 0;
    unsigned long max_abs_diff_row = 0;
    unsigned int ncols = table1->get_number_of_columns();
    unsigned long nrows = table1->get_number_of_rows();
    for (unsigned int col = 0; col < ncols; ++col)
    {
      const_p_teca_variant_array col1 = table1->get_column(col);
      const_p_teca_variant_array col2 = table2->get_column(col);

      for (unsigned long row = 0; row < nrows; ++row)
      {
        double v1, v2;
        col1->get(row, v1);
        col1->get(row, v2);
        double abs_diff = std::abs(v1 - v2);
        if (abs_diff > max_abs)
        {
          max_abs = abs_diff;
          max_abs_diff_col = col;
          max_abs_diff_row = row;
          if (max_abs > this->tolerance)
            break;
        }
      }
    }

    if (max_abs > this->tolerance)
    {
      std::cout << "Datasets differ. Max absolute difference " 
                << max_abs 
                << " exceeds tolerance "
                << this->tolerance 
                << " at row "
                << max_abs_diff_row 
                << ", column " 
                << max_abs_diff_col
                << "."
                << std::endl;
    }
    else
    {
      std::cout << "Datasets are the same." << std::endl;
    }
    return 0;
}

// --------------------------------------------------------------------------
int teca_dataset_diff::compare_cartesian_meshes(
    const_p_teca_cartesian_mesh mesh1,
    const_p_teca_cartesian_mesh mesh2)
{
    // If the meshes are different sizes, the datasets differ.

    // If the collections of arrays are different, the datasets differ.
 
    // If the elementwise differences exceed the tolerance, the datasets differ.
    return 0;
}

