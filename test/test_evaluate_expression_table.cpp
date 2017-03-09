#include "teca_config.h"
#include "teca_table_reader.h"
#include "teca_table_to_stream.h"
#include "teca_evaluate_expression.h"
#include "teca_table_writer.h"
#include "teca_dataset_diff.h"
#include "teca_file_util.h"
#include "teca_system_interface.h"

#include <vector>
#include <string>
#include <iostream>

using std::cerr;
using std::endl;

int main(int argc, char **argv)
{
    teca_system_interface::set_stack_trace_on_error();

    // parse command line
    if (argc < 4)
    {
        cerr << endl << "Usage error:" << endl
            << "test_evaluate_expression_table [input table] [test baseline] [expression]"
            << endl << endl;
        return -1;
    }
    std::string input_table = argv[1];
    std::string baseline_table = argv[2];
    std::string expression = argv[3];

    // create the pipeline
    p_teca_table_reader input_reader = teca_table_reader::New();
    input_reader->set_file_name(input_table);

    p_teca_evaluate_expression eval_expr = teca_evaluate_expression::New();
    eval_expr->set_input_connection(input_reader->get_output_port());
    eval_expr->set_expression(expression);
    eval_expr->set_result_variable("test_tracks");

    if (teca_file_util::file_exists(baseline_table.c_str()))
    {
        // run the test
        p_teca_table_reader baseline_table_reader = teca_table_reader::New();
        baseline_table_reader->set_file_name(baseline_table);

        p_teca_dataset_diff diff = teca_dataset_diff::New();
        diff->set_input_connection(0, baseline_table_reader->get_output_port());
        diff->set_input_connection(1, eval_expr->get_output_port());
        diff->update();
    }
    else
    {
        // make a baseline
        cerr << "generating baseline image " << baseline_table << endl;

        /*p_teca_table_to_stream dump_table = teca_table_to_stream::New();
        dump_table->set_input_connection(eval_expr->get_output_port());*/

        p_teca_table_writer table_writer = teca_table_writer::New();
        //table_writer->set_input_connection(dump_table->get_output_port());
        table_writer->set_input_connection(eval_expr->get_output_port());
        table_writer->set_file_name(baseline_table);
        table_writer->update();
        return -1;
    }

    return 0;
}
