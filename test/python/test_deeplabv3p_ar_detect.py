try:
    from mpi4py import *
    rank = MPI.COMM_WORLD.Get_rank()
    n_ranks = MPI.COMM_WORLD.Get_size()
except ImportError:
    rank = 0
    n_ranks = 1
import os
import sys
import numpy as np
from teca import *


def main():
    set_stack_trace_on_error()
    set_stack_trace_on_mpi_error()

    if (len(sys.argv) != 7):
        sys.stderr.write(
            "\n\nUsage error:\n"
            "test_deeplabv3p_ar_detect [deeplab model] [resnet model]"
            "[mesh data regex] [baseline mesh]"
            "[water vapor var] [num threads]\n\n"
            )
        sys.exit(-1)

    # parse command line
    deeplab_model = sys.argv[1]
    resnet_model = sys.argv[2]
    mesh_data_regex = sys.argv[3]
    baseline_mesh = sys.argv[4]
    water_vapor_var = sys.argv[5]
    n_threads = int(sys.argv[6])

    cf_reader = teca_cf_reader.New()
    cf_reader.set_files_regex(mesh_data_regex)
    cf_reader.set_t_axis_variable("")
    cf_reader.set_periodic_in_x(1)

    deeplabv3p_ar_detect = teca_deeplabv3p_ar_detect.New()
    deeplabv3p_ar_detect.set_input_connection(
        cf_reader.get_output_port()
        )
    deeplabv3p_ar_detect.set_variable_name(water_vapor_var)
    deeplabv3p_ar_detect.set_num_threads(n_threads)

    deeplabv3p_ar_detect.build_model(
        deeplab_model,
        resnet_model
        )

    if os.path.exists(baseline_mesh):
        # run the test
        baseline_mesh_reader = teca_cartesian_mesh_reader.New()
        baseline_mesh_reader.set_file_name(baseline_mesh)

        diff = teca_dataset_diff.New()
        diff.set_input_connection(0, baseline_mesh_reader.get_output_port())
        diff.set_input_connection(1, deeplabv3p_ar_detect.get_output_port())
        diff.set_tolerance(1e-4)
        diff.update()
    else:
        # make a baseline
        if rank == 0:
            sys.stdout.write(
                'generating baseline image ' + baseline_mesh + '\n'
                )
        wri = teca_cartesian_mesh_writer.New()
        wri.set_input_connection(deeplabv3p_ar_detect.get_output_port())
        wri.set_file_name(baseline_mesh)

        wri.update()
        sys.exit(-1)


if __name__ == '__main__':
    main()
