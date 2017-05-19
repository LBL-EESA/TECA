#include "teca_config.h"
#include "teca_table_reader.h"
#include "teca_table_to_stream.h"
#include "teca_evaluate_expression.h"
#include "teca_table_region_mask.h"
#include "teca_table_remove_rows.h"
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
    if (argc < 3)
    {
        cerr << endl << "Usage error:" << endl
            << "test_event_filter [input] [output]"
            << endl << endl;
        return -1;
    }

    std::string table_in = argv[1];
    std::string baseline = argv[2];

    p_teca_table_reader tr = teca_table_reader::New();
    tr->set_file_name(table_in);

    p_teca_table_region_mask rm = teca_table_region_mask::New();
    rm->set_input_connection(tr->get_output_port());
    rm->set_x_coordinate_column("lon");
    rm->set_y_coordinate_column("lat");
    rm->set_region_x_coordinates({180, 180, 270, 270, 180});
    rm->set_region_y_coordinates({-10, 10, 10, -10, -10});
    rm->set_region_sizes({5});
    rm->set_result_column("in_spatial");

    p_teca_evaluate_expression ee = teca_evaluate_expression::New();
    ee->set_input_connection(rm->get_output_port());
    ee->set_expression("((time > 4196.23) && (time < 4196.39))");
    ee->set_result_variable("in_temporal");

    p_teca_table_remove_rows rr = teca_table_remove_rows::New();
    rr->set_input_connection(ee->get_output_port());
    rr->set_mask_expression("!(in_temporal && in_spatial)");
    rr->set_remove_dependent_variables(1);

    if (teca_file_util::file_exists(baseline.c_str()))
    {
        // run the test
        p_teca_table_reader baseline_table_reader = teca_table_reader::New();
        baseline_table_reader->set_file_name(baseline);

        p_teca_dataset_diff diff = teca_dataset_diff::New();
        diff->set_input_connection(0, baseline_table_reader->get_output_port());
        diff->set_input_connection(1, rr->get_output_port());
        diff->update();
    }
    else
    {
        // make a baseline
        cerr << "generating baseline image " << baseline << endl;

        /*p_teca_table_to_stream dump_table = teca_table_to_stream::New();
        dump_table->set_input_connection(eval_expr->get_output_port());*/

        p_teca_table_writer table_writer = teca_table_writer::New();
        //table_writer->set_input_connection(dump_table->get_output_port());
        table_writer->set_input_connection(rr->get_output_port());
        table_writer->set_file_name(baseline);
        table_writer->update();
        return -1;
    }

    return 0;
}
