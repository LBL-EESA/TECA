#ifndef create_test_table_h
#define create_test_table_h

#include "teca_table.h"

// This creates a TECA table containing some basic test data that is used by 
// the TECA table reader/writer tests and the dataset_diff test.
const_p_teca_dataset create_test_table(long step);

#endif
