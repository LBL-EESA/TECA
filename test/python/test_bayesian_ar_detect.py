try:
    from mpi4py import *
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

if (len(sys.argv) != 9):
    sys.stderr.write('\n\nUsage error:\n'\
        'test_bayesian_ar_detect [parameter table] [mesh data regex] ' \
        '[baseline table] [water vapor var] [out file] [num threads] ' \
        '[first step] [last step]\n\n')
    sys.exit(-1)

# parse command line
parameter_table = sys.argv[1]
mesh_data_regex = sys.argv[2]
baseline_table = sys.argv[3]
water_vapor_var = sys.argv[4]
out_file_name = sys.argv[5]
n_threads = int(sys.argv[6])
first_step =  int(sys.argv[7])
last_step = int(sys.argv[8])

if (rank == 0):
    sys.stderr.write('Testing on %d MPI processes %d threads\n'%(n_ranks, n_threads))

# create the pipeline
parameter_reader = teca_table_reader.New()
parameter_reader.set_file_name(parameter_table)

mesh_data_reader = teca_cf_reader.New()
mesh_data_reader.set_files_regex(mesh_data_regex)
mesh_data_reader.set_periodic_in_x(1)

ar_detect = teca_bayesian_ar_detect.New()
ar_detect.set_input_connection(0, parameter_reader.get_output_port())
ar_detect.set_input_connection(1, mesh_data_reader.get_output_port())
ar_detect.set_water_vapor_variable(water_vapor_var)
ar_detect.set_thread_pool_size(n_threads)

seg = teca_binary_segmentation.New()
seg.set_input_connection(ar_detect.get_output_port())
seg.set_low_threshold_value(0.25)
seg.set_threshold_variable('ar_probability')
seg.set_segmentation_variable('ar_probability_0.25')

cc = teca_connected_components.New()
cc.set_input_connection(seg.get_output_port())
cc.set_segmentation_variable('ar_probability_0.25')
cc.set_component_variable('ars')

ca = teca_2d_component_area.New()
ca.set_input_connection(cc.get_output_port())
ca.set_component_variable('ars')

wri = teca_cartesian_mesh_writer.New()
wri.set_input_connection(ca.get_output_port())
wri.set_file_name(out_file_name)

cs = teca_component_statistics.New()
cs.set_input_connection(wri.get_output_port())

map_reduce = teca_table_reduce.New()
map_reduce.set_input_connection(cs.get_output_port())
map_reduce.set_start_index(first_step)
map_reduce.set_end_index(last_step)
map_reduce.set_verbose(1)
map_reduce.set_thread_pool_size(1)

# sort results in time
sort = teca_table_sort.New()
sort.set_input_connection(map_reduce.get_output_port())
sort.set_index_column('global_component_ids')

if os.path.exists(baseline_table):
    # run the test
    baseline_table_reader = teca_table_reader.New()
    baseline_table_reader.set_file_name(baseline_table)

    diff = teca_dataset_diff.New()
    diff.set_input_connection(0, baseline_table_reader.get_output_port())
    diff.set_input_connection(1, sort.get_output_port())
    diff.update()
else:
    # make a baseline
    if rank == 0:
        cerr << 'generating baseline image ' << baseline_table << endl

    tts = teca_table_to_stream.New()
    tts.set_input_connection(sort.get_output_port())

    table_writer = teca_table_writer.New()
    table_writer.set_input_connection(tts.get_output_port())
    table_writer.set_file_name(baseline_table)

    table_writer.update()
    sys.exit(-1)
