#include "teca_config.h"
#include "teca_table_writer.h"
#include "teca_table_reader.h"
#include "teca_table.h"
#include "teca_dataset_diff.h"
#include "teca_index_executive.h"
#include "teca_test_util.h"
#include "teca_system_interface.h"

#include <iostream>
using namespace std;

int main(int, char **)
{
    teca_system_interface::set_stack_trace_on_error();

    // Write a test table.
    p_teca_algorithm s = teca_test_util::test_table_server::New(1);

    p_teca_table_writer w = teca_table_writer::New();
    w->set_input_connection(s->get_output_port());
    w->set_file_name("table_reader_test.bin");
    w->set_executive(teca_index_executive::New());

    w->update();

    // Set up reader to read it back in
    p_teca_table_reader r = teca_table_reader::New();
    r->set_file_name("table_reader_test.bin");

    // create the same table in memory
    s = teca_test_util::test_table_server::New(1);

    // Set up the dataset diff algorithm
    p_teca_dataset_diff diff = teca_dataset_diff::New();
    diff->set_input_connection(0, s->get_output_port());
    diff->set_input_connection(1, r->get_output_port());

    // run the test
    diff->update();

    return 0;
}
