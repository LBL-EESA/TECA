#include "teca_cf_reader.h"
#include "teca_normalize_coordinates.h"
#include "teca_indexed_dataset_cache.h"
#include "teca_simple_moving_average.h"
#include "teca_cartesian_mesh_writer.h"
#include "teca_cf_writer.h"
#include "teca_dataset_diff.h"
#include "teca_index_executive.h"
#include "teca_system_interface.h"
#include "teca_system_util.h"

#include <vector>
#include <string>
#include <iostream>
using namespace std;

int main(int argc, char **argv)
{
    teca_system_interface::set_stack_trace_on_error();

    if (argc < 8)
    {
        std::cerr << std::endl << "Usage error:" << std::endl
            << "test_simple_moving_average [input regex] [baseline] [first step] [last step]"
               "  [filter width] [n threads] [array] [array] ..." << std::endl
            << std::endl;
        return -1;
    }

    // parse command line
    string regex = argv[1];
    string baseline = argv[2];
    long first_step = atoi(argv[3]);
    long last_step = atoi(argv[4]);
    int filter_width = atoi(argv[5]);
    int n_threads = atoi(argv[6]);
    std::vector<std::string> arrays;
    arrays.push_back(argv[7]);
    for (int i = 8; i < argc; ++i)
        arrays.push_back(argv[i]);

    // create the cf reader
    p_teca_cf_reader r = teca_cf_reader::New();
    r->set_files_regex(regex);

    // normalize coords
    p_teca_normalize_coordinates c = teca_normalize_coordinates::New();
    c->set_input_connection(r->get_output_port());

    // ds cache
    p_teca_indexed_dataset_cache dsc = teca_indexed_dataset_cache::New();
    dsc->set_input_connection(c->get_output_port());
    dsc->set_max_cache_size(2*n_threads*filter_width);

    // temporal avg
    p_teca_simple_moving_average a = teca_simple_moving_average::New();
    a->set_filter_width(filter_width);
    a->set_filter_type(teca_simple_moving_average::backward);
    a->set_input_connection(dsc->get_output_port());

    bool do_test = true;
    teca_system_util::get_environment_variable("TECA_DO_TEST", do_test);
    if (do_test)
    {
        std::cerr << "running the test..." << std::endl;

        baseline += ".*\\.nc$";

        p_teca_cf_reader br = teca_cf_reader::New();
        br->set_files_regex(baseline);

        // executive
        p_teca_index_executive rex = teca_index_executive::New();
        rex->set_start_index(first_step);
        rex->set_end_index(last_step);
        rex->set_arrays(arrays);

        p_teca_dataset_diff diff = teca_dataset_diff::New();
        diff->set_input_connection(0, br->get_output_port());
        diff->set_input_connection(1, a->get_output_port());
        diff->set_verbose(1);
        diff->set_executive(rex);
        // TODO : test with threads
        //diff->set_thread_pool_size(n_threads);
        diff->update();
    }
    else
    {
        std::cerr << "writing the baseline..." << std::endl;

        baseline += "_%t%.nc";

        // writer
        p_teca_cf_writer w = teca_cf_writer::New();
        w->set_input_connection(a->get_output_port());
        w->set_thread_pool_size(n_threads);
        w->set_point_arrays(arrays);
        w->set_file_name(baseline);
        w->set_first_step(first_step);
        w->set_last_step(last_step);
        w->set_steps_per_file(10000);
        w->update();
    }

    return 0;
}
