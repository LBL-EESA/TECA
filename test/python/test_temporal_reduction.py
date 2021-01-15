try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    n_ranks = MPI.COMM_WORLD.Get_size()
except:
    rank = 0
    n_ranks = 1
from teca import *
import numpy as np
import sys
import os

set_stack_trace_on_error()
set_stack_trace_on_mpi_error()

argc = len(sys.argv)
if argc < 9:
    sys.stderr.write('usage: app [in file regex] [z axis var] [out file base] '
                     '[steps per file] [n threads] [interval] [operator] '
                     '[use fill value] [array name 0] ... [array name n]\n')
    sys.exit(-1)

files = sys.argv[1]
z_axis = '' if sys.argv[2] == '.' else sys.argv[2]
out_base = sys.argv[3]
steps_per_file = int(sys.argv[4])
n_threads = int(sys.argv[5])
interval = sys.argv[6]
operator = sys.argv[7]
use_fill = int(sys.argv[8])
arrays = sys.argv[9:]

if rank == 0:
    sys.stderr.write('testing on %d ranks'%(n_ranks))
    sys.stderr.write('n_threads=%d\n'%(n_threads))
    sys.stderr.write('steps_per_file=%d\n'%(steps_per_file))
    sys.stderr.write('interval=%s\n'%(interval))
    sys.stderr.write('operator=%s\n'%(operator))
    sys.stderr.write('arrays=%s\n'%(str(arrays)))
    sys.stderr.write('use_fill=%d\n'%(use_fill))

cfr = teca_cf_reader.New()
cfr.set_files_regex(files)
cfr.set_z_axis_variable(z_axis)

mav = teca_temporal_reduction.New()
mav.set_input_connection(cfr.get_output_port())
mav.set_interval(interval)
mav.set_operator(operator)
mav.set_point_arrays(arrays)
mav.set_use_fill_value(use_fill)
mav.set_verbose(2)
mav.set_thread_pool_size(n_threads)
mav.set_stream_size(2)

do_test = 1
if do_test:
    # run the test
    if rank == 0:
        sys.stderr.write('running test...\n')
    bcfr = teca_cf_reader.New()
    bcfr.set_files_regex(('%s_%s_%s_.*\\.nc$'%(out_base,interval,operator)))
    bcfr.set_z_axis_variable(z_axis)

    exe = teca_index_executive.New()
    exe.set_arrays(arrays)

    diff = teca_dataset_diff.New()
    diff.set_input_connection(0, bcfr.get_output_port())
    diff.set_input_connection(1, mav.get_output_port())
    diff.set_relative_tolerance(1e-5)
    diff.set_executive(exe)

    diff.update()
else:
    # make a baseline
    if rank == 0:
        sys.stderr.write('generating baseline...\n')
    cfw = teca_cf_writer.New()
    cfw.set_input_connection(mav.get_output_port())
    cfw.set_verbose(1)
    cfw.set_thread_pool_size(1)
    cfw.set_steps_per_file(steps_per_file)
    cfw.set_file_name('%s_%s_%s_%%t%%.nc'%(out_base,interval,operator))
    cfw.set_point_arrays(arrays)
    cfw.update()
