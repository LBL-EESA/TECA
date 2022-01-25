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


class custom_reductions:

    def New(self, op_name):
        """ factory method """

        if op_name == 'sum':
            return self.sum()

        raise RuntimeError('Invalid reduction operator %s' % (op_name))

    class sum:
        """ sum reduction operator """
        def __init__(self):
            self.fill_value = None

        def initialize(self, fill_value):
            self.fill_value = fill_value

        def update(self, dev, out_array, out_valid, in_array, in_valid):

            res = out_array + in_array
            res_valid = None

            if self.fill_value is not None:
                res_valid = np.ones_like(out_array, dtype=np.int8)

                in_bad = np.logical_not(in_valid)
                res[in_bad] = self.fill_value
                res_valid[out_bad] = np.int8(0)

                out_bad = np.logical_not(out_valid)
                res[out_bad] = self.fill_value
                res_valid[in_bad] = np.int8(0)

            return res, out_valid

        def finalize(self, dev, out_array, out_valid):
            return out_array, out_valid


class custom_intervals:

    def New(self, it_name, t, units, calendar):
        """ factory method """

        if it_name == 'five_steps':
            return self.five_steps(t, units, calendar)

        raise RuntimeError('Invalid interval iterator %s' % (it_name))


    class five_steps:
        """ iterates in intervals of 5 steps """

        def __init__(self, t, units, calendar):
            self.time = t
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):

            i0 = self.index
            i1 = self.index + 4

            if i1 >= len(self.time):
                raise StopIteration

            self.index += 5

            return teca_temporal_reduction. \
                time_interval(self.time[i0], i0, i1)




argc = len(sys.argv)
if argc < 6:
    sys.stderr.write('usage: app [in file regex] [z axis var] [out file base] '
                     '[use fill value] [array name 0] ... [array name n]\n')
    sys.exit(-1)

files = sys.argv[1]
z_axis = '' if sys.argv[2] == '.' else sys.argv[2]
out_base = sys.argv[3]
interval = 'five_steps'
operator = 'sum'
use_fill = int(sys.argv[4])
arrays = sys.argv[5:]

if rank == 0:
    sys.stderr.write('testing on %d ranks'%(n_ranks))
    sys.stderr.write('interval=%s\n'%(interval))
    sys.stderr.write('operator=%s\n'%(operator))
    sys.stderr.write('arrays=%s\n'%(str(arrays)))
    sys.stderr.write('use_fill=%d\n'%(use_fill))

cfr = teca_cf_reader.New()
cfr.set_files_regex(files)
cfr.set_z_axis_variable(z_axis)

mav = teca_temporal_reduction.New()
mav.set_input_connection(cfr.get_output_port())
mav.set_interval_iterator_factory(custom_intervals())
mav.set_reduction_operator_factory(custom_reductions())
mav.set_interval(interval)
mav.set_operator(operator)
mav.set_point_arrays(arrays)
mav.set_use_fill_value(use_fill)
mav.set_verbose(1)
mav.set_thread_pool_size(1)
mav.set_stream_size(2)

do_test = 1
if do_test:
    # run the test
    if rank == 0:
        sys.stderr.write('running test...\n')
        sys.stderr.write('regex=%s_%s_%s_.*\\.nc$\n'%(out_base,interval,operator))
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
        sys.stderr.write('filename=%s_%s_%s_%%t%%.nc\n'%(out_base,interval,operator))
    cfw = teca_cf_writer.New()
    cfw.set_input_connection(mav.get_output_port())
    cfw.set_verbose(1)
    cfw.set_thread_pool_size(1)
    cfw.set_layout('yearly')
    cfw.set_steps_per_file(steps_per_file)
    cfw.set_file_name('%s_%s_%s_%%t%%.nc'%(out_base,interval,operator))
    cfw.set_point_arrays(arrays)
    cfw.update()
