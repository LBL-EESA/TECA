#include "teca_config.h"
#include "teca_table_reader.h"
#include "teca_table_writer.h"
#include "teca_dataset_diff.h"
#include "teca_index_executive.h"
#include "teca_table.h"
#include "teca_test_util.h"
#include "teca_system_interface.h"
#include "teca_common.h"

#include <iostream>
using namespace std;

int main(int argc, char **argv)
{
    teca_error::set_error_message_handler();
    teca_system_interface::set_stack_trace_on_error();

    if (argc < 2)
    {
        std::cerr << "Error: test_dataset_diff [compare same]" << std::endl;
        return -1;
    }

    int compare_same = atoi(argv[1]);

    // write 2 test files.
    p_teca_algorithm s = teca_test_util::test_table_server::New(2);

    p_teca_table_writer w = teca_table_writer::New();
    w->set_input_connection(s->get_output_port());
    w->set_executive(teca_index_executive::New());
    w->set_file_name("dataset_diff_test_%t%.%e%");
    w->set_output_format_bin();

    w->update();


    // read two file and compare them. if compare_same is true then
    // read the first file on both readers, if not read both files
    p_teca_table_reader r1 = teca_table_reader::New();
    p_teca_table_reader r2 = teca_table_reader::New();

    if (compare_same)
    {
        // these datasets are the same, hence diff should pass
        r1->set_file_name("dataset_diff_test_000000.bin");
        r2->set_file_name("dataset_diff_test_000000.bin");
    }
    else
    {
        // these datasets are different, hence diff should fail
        r1->set_file_name("dataset_diff_test_000000.bin");
        r2->set_file_name("dataset_diff_test_000001.bin");
    }

    // diff the output of the readers
    p_teca_dataset_diff diff = teca_dataset_diff::New();
    diff->set_input_connection(0, r1->get_output_port());
    diff->set_input_connection(1, r2->get_output_port());

    diff->update();

    return 0;
}
