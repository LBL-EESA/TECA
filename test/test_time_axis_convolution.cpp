#include "teca_cf_reader.h"
#include "teca_normalize_coordinates.h"
#include "teca_indexed_dataset_cache.h"
#include "teca_time_axis_convolution.h"
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

    if (argc < 10)
    {
        std::cerr << std::endl << "Usage error:" << std::endl
            << "test_time_axis_convolution [input regex] [baseline]"
               " [kernel type] [kernel width] [stencil type] [high_pass]"
               " [first step] [last step] [n threads] [array]"
               " [array] ..." << std::endl
            << std::endl;
        return -1;
    }

    // parse command line
    std::string regex = argv[1];
    std::string baseline = argv[2];
    std::string kernel_type = argv[3];
    int kernel_width = atoi(argv[4]);
    std::string stencil_type = argv[5];
    int use_high_pass = atoi(argv[6]);
    long first_step = atoi(argv[7]);
    long last_step = atoi(argv[8]);
    int n_threads = atoi(argv[9]);
    std::vector<std::string> in_arrays;
    in_arrays.push_back(argv[10]);
    for (int i = 11; i < argc; ++i)
        in_arrays.push_back(argv[i]);

    // create the cf reader
    p_teca_cf_reader reader = teca_cf_reader::New();
    reader->set_files_regex(regex);

    // normalize coords
    p_teca_normalize_coordinates coords = teca_normalize_coordinates::New();
    coords->set_input_connection(reader->get_output_port());

    // ds cache
    p_teca_indexed_dataset_cache dsc = teca_indexed_dataset_cache::New();
    dsc->set_input_connection(coords->get_output_port());
    dsc->set_max_cache_size(2*n_threads*kernel_width);

    // time axis convolution
    p_teca_time_axis_convolution conv = teca_time_axis_convolution::New();
    conv->set_input_connection(dsc->get_output_port());
    conv->set_stencil_type(stencil_type);
    conv->set_kernel_name(kernel_type);
    conv->set_kernel_width(kernel_width);
    conv->set_use_high_pass(use_high_pass);

    int n_arrays = in_arrays.size();
    std::vector<std::string> arrays(2*n_arrays);
    for (int i = 0; i < n_arrays; ++i)
    {
        int ii = 2*i;
        arrays[ii] = in_arrays[i];
        arrays[ii+1] = in_arrays[i] + conv->get_variable_postfix();
    }

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
        diff->set_input_connection(1, conv->get_output_port());
        diff->set_verbose(1);
        diff->set_executive(rex);
        //diff->set_thread_pool_size(n_threads);
        diff->update();
    }
    else
    {
        std::cerr << "writing the baseline..." << std::endl;

        baseline += "_%t%.nc";

        // writer
        p_teca_cf_writer writer = teca_cf_writer::New();
        writer->set_input_connection(conv->get_output_port());
        writer->set_thread_pool_size(n_threads);
        writer->set_point_arrays(arrays);
        writer->set_file_name(baseline);
        writer->set_first_step(first_step);
        writer->set_last_step(last_step);
        writer->set_layout_to_yearly();

        writer->update();
    }

    return 0;
}
