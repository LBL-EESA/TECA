#include "teca_config.h"
#include "teca_metadata.h"
#include "teca_cf_reader.h"
#include "teca_temporal_reduction.h"
#include "teca_cf_writer.h"
#include "teca_mpi_manager.h"
#include "teca_system_interface.h"

#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <cstring>
#include <chrono>

using microseconds_t = std::chrono::duration<double, std::chrono::microseconds::period>;

int main(int argc, char **argv)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    teca_mpi_manager mpi_man(argc, argv);
    int rank = mpi_man.get_comm_rank();
    int n_ranks = mpi_man.get_comm_size();

    teca_system_interface::set_stack_trace_on_error();
    teca_system_interface::set_stack_trace_on_mpi_error();

    if (argc != 25)
    {
        if (rank == 0)
        {
            std::cerr << "test_temporal_reduction [files regex]"
                " [x axis var] [y axis var] [z axis var] [t axis var]"
                " [red var] [first step] [last step]"
                " [red threads] [red threads per dev] [red ranks per dev] [red stream size] [red bind threads] [red prop dev]"
                " [reduction interval] [reduction operator] [steps per request]"
                " [out file] [file layout]"
                " [wri threads] [wri threads per dev] [wri ranks per dev] [wri stream size] [wri bind threads]"
                << std::endl;
        }
        return -1;
    }

    std::string files_regex = argv[1];
    std::string x_axis_var = argv[2];
    std::string y_axis_var = argv[3];
    std::string z_axis_var = argv[4];
    std::string t_axis_var = argv[5];
    std::string red_var = argv[6];
    int first_step = atoi(argv[7]);
    int last_step = atoi(argv[8]);

    int n_red_threads = atoi(argv[9]);
    int red_threads_per_dev = atoi(argv[10]);
    int red_ranks_per_dev = atoi(argv[11]);
    int red_stream_size = atoi(argv[12]);
    int red_bind_threads = atoi(argv[13]);
    int red_prop_dev_id = atoi(argv[14]);

    std::string red_int = argv[15];
    std::string red_op = argv[16];
    int steps_per_req = atoi(argv[17]);

    std::string ofile_name = argv[18];
    std::string layout = argv[19];
    int n_wri_threads = atoi(argv[20]);
    int wri_threads_per_dev = atoi(argv[21]);
    int wri_ranks_per_dev = atoi(argv[22]);
    int wri_stream_size = atoi(argv[23]);
    int wri_bind_threads = atoi(argv[24]);


    if (rank == 0)
    {
        std::cerr << "n_ranks=" << n_ranks
            << "  n_red_threads=" << n_red_threads << "  red_threads_per_dev=" << red_threads_per_dev
            << "  red_ranks_per_dev=" << red_ranks_per_dev << "red_stream_size=" << red_stream_size
            << "  red_bind_threads=" << red_bind_threads << "  red_pro_dev_id=" << red_prop_dev_id
            << "  red_int=" << red_int << "  red_op=" << red_op << "  steps_per_req=" << steps_per_req
            << "  layout=" << layout << "  n_wri_threads=" << n_wri_threads
            << "  wri_threads_per_dev=" << wri_threads_per_dev
            << "  wri_ranks_per_dev=" << wri_ranks_per_dev << "  wri_stream_size=" << wri_stream_size
            << "  wri_bind_threads=" << wri_bind_threads
            << std::endl;
    }

    // reader
    auto cf_reader = teca_cf_reader::New();
    cf_reader->set_x_axis_variable(x_axis_var);
    cf_reader->set_y_axis_variable(y_axis_var);
    cf_reader->set_z_axis_variable(z_axis_var == "." ? std::string() : z_axis_var);
    cf_reader->set_t_axis_variable(t_axis_var);
    cf_reader->set_files_regex(files_regex);

    // temporal reduction
    auto reduc = teca_cpp_temporal_reduction::New();
    reduc->set_input_connection(cf_reader->get_output_port());
    reduc->set_verbose(1);
    reduc->set_threads_per_device(red_threads_per_dev);
    reduc->set_ranks_per_device(red_ranks_per_dev);
    reduc->set_bind_threads(red_bind_threads);
    reduc->set_stream_size(red_stream_size);
    reduc->set_propagate_device_assignment(red_prop_dev_id);
    reduc->set_thread_pool_size(n_red_threads);
    reduc->set_interval(red_int);
    reduc->set_operation(red_op);
    reduc->set_point_arrays({red_var});
    reduc->set_steps_per_request(steps_per_req);

    // writer
    auto cfw = teca_cf_writer::New();
    cfw->set_input_connection(reduc->get_output_port());
    cfw->set_verbose(1);
    cfw->set_threads_per_device(wri_threads_per_dev);
    cfw->set_ranks_per_device(wri_ranks_per_dev);
    cfw->set_stream_size(wri_stream_size);
    cfw->set_bind_threads(wri_bind_threads);
    cfw->set_thread_pool_size(n_wri_threads);
    cfw->set_file_name(ofile_name);
    cfw->set_layout(layout);
    cfw->set_point_arrays({red_var});
    cfw->set_first_step(first_step);
    cfw->set_last_step(last_step);

    cfw->update();

    if (rank == 0)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        microseconds_t dt(t1 - t0);
        std::cerr << "total runtime : " << (dt.count() / 1e6) << std::endl;
    }

    return 0;
}
