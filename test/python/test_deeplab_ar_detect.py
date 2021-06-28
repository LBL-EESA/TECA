try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    n_ranks = MPI.COMM_WORLD.Get_size()
except ImportError:
    rank = 0
    n_ranks = 1
import os
import sys
import numpy as np
from teca import *


set_stack_trace_on_error()
set_stack_trace_on_mpi_error()

if (len(sys.argv) != 6):
    sys.stderr.write('\n\nUsage error:\n'
                     'test_deeplab_ar_detect [deeplab model] '
                     '[mesh data regex] [baseline] '
                     '[water vapor var] [num threads]\n\n')
    sys.exit(-1)

# parse command line
deeplab_model = sys.argv[1]
input_regex = sys.argv[2]
baseline = sys.argv[3]
water_vapor_var = sys.argv[4]
n_threads = int(sys.argv[5])
vrb = 1
dev = 'cuda'

cf_reader = teca_cf_reader.New()
cf_reader.set_files_regex(input_regex)
cf_reader.set_periodic_in_x(1)

ar_detect = teca_deeplab_ar_detect.New()
ar_detect.set_input_connection(cf_reader.get_output_port())
ar_detect.set_target_device(dev)
ar_detect.set_verbose(vrb)
ar_detect.set_ivt_variable(water_vapor_var)
ar_detect.set_thread_pool_size(n_threads)
ar_detect.load_model(deeplab_model)

rex = teca_index_executive.New()
rex.set_arrays(['ar_probability'])
rex.set_verbose(vrb)

do_test = 1
if do_test:
    # run the test
    if rank == 0:
        sys.stdout.write('running test...\n')

    baseline_reader = teca_cf_reader.New()
    baseline_reader.set_files_regex('%s-.*\\.nc'%(baseline))
    baseline_reader.set_periodic_in_x(1)

    diff = teca_dataset_diff.New()
    diff.set_input_connection(0, baseline_reader.get_output_port())
    diff.set_input_connection(1, ar_detect.get_output_port())
    diff.set_relative_tolerance(5e-4)
    diff.set_absolute_tolerance(1e-6)
    diff.set_executive(rex)
    diff.update()

else:
    # make a baseline
    if rank == 0:
        sys.stdout.write('generating baseline %s...\n'%(baseline))

    baseline_writer = teca_cf_writer.New()
    baseline_writer.set_input_connection(ar_detect.get_output_port())
    baseline_writer.set_file_name('%s-%%t%%.nc'%(baseline))
    baseline_writer.set_point_arrays(['ar_probability'])
    baseline_writer.set_thread_pool_size(1)

    baseline_writer.update()
    sys.exit(-1)
