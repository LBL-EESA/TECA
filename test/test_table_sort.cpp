#include "teca_config.h"
#include "teca_table_reader.h"
#include "teca_table_to_stream.h"
#include "teca_table_sort.h"
#include "teca_table_writer.h"
#include "teca_dataset_diff.h"
#include "teca_file_util.h"
#include "teca_system_interface.h"

#include <vector>
#include <string>
#include <iostream>
using namespace std;

int main(int argc, char **argv)
{
    teca_system_interface::set_stack_trace_on_error();

    // parse command line
    if (argc < 4)
    {
        cerr << endl << "Usage error:" << endl
            << "test_table_sort [input table] [index column] [test baseline]"
            << endl << endl;
        return -1;
    }
    string input_table = argv[1];
    string baseline_table = argv[2];
    string index_col = argv[3];

    // create the pipeline
    p_teca_table_reader input_reader = teca_table_reader::New();
    input_reader->set_file_name(input_table);

    p_teca_table_sort sort = teca_table_sort::New();
    sort->set_input_connection(input_reader->get_output_port());
    sort->set_index_column("step");

    if (teca_file_util::file_exists(baseline_table.c_str()))
    {
        // run the test
        p_teca_table_reader baseline_table_reader = teca_table_reader::New();
        baseline_table_reader->set_file_name(baseline_table);

        p_teca_dataset_diff diff = teca_dataset_diff::New();
        diff->set_input_connection(0, baseline_table_reader->get_output_port());
        diff->set_input_connection(1, sort->get_output_port());
        diff->update();
    }
    else
    {
        // make a baseline
        cerr << "generating baseline image " << baseline_table << endl;

        p_teca_table_to_stream post_sort = teca_table_to_stream::New();
        post_sort->set_input_connection(sort->get_output_port());

        p_teca_table_writer table_writer = teca_table_writer::New();
        table_writer->set_input_connection(post_sort->get_output_port());
        table_writer->set_file_name(baseline_table);
        table_writer->set_output_format_bin();
        table_writer->update();
        return -1;
    }

    return 0;
}
