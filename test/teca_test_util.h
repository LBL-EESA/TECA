#ifndef create_test_table_h
#define create_test_table_h

#include "teca_table.h"

namespace test_util
{
// This creates a TECA table containing some basic test data that
// is used by the TECA table reader/writer tests and the dataset_diff
// test.
enum {base_table,
    break_string_col,
    break_int_col,
    break_float_col
    };
const_p_teca_dataset create_test_table(long step,
    int tid=test_util::base_table);
}

#endif
