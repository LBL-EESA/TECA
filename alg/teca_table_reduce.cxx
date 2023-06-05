#include "teca_table_reduce.h"
#include "teca_table.h"

#include <iostream>
#include <limits>

using std::cerr;
using std::endl;
using std::vector;

// --------------------------------------------------------------------------
teca_table_reduce::teca_table_reduce()
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
p_teca_dataset teca_table_reduce::reduce(int device_id,
    const const_p_teca_dataset &left_ds, const const_p_teca_dataset &right_ds)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_table_reduce::reduce" << endl;
#endif
     (void) device_id;

    const_p_teca_table left_table
        = std::dynamic_pointer_cast<const teca_table>(left_ds);

    const_p_teca_table right_table
        = std::dynamic_pointer_cast<const teca_table>(right_ds);

    p_teca_table output_table;

    bool have_left = left_table && *left_table;
    bool have_right = right_table && *right_table;

    if (have_left && have_right)
    {
        output_table
            = std::dynamic_pointer_cast<teca_table>(left_table->new_copy());

        output_table->concatenate_rows(right_table);
    }
    else
    if (have_left)
    {
        output_table
            = std::dynamic_pointer_cast<teca_table>(left_table->new_copy());
    }
    else
    if (have_right)
    {
        output_table
            = std::dynamic_pointer_cast<teca_table>(right_table->new_copy());
    }

    return output_table;
}
