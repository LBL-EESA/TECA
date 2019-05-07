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
import matplotlib as mpl
mpl.use('Agg')

set_stack_trace_on_error()
set_stack_trace_on_mpi_error()

if len(sys.argv) != 6:
    sys.stderr.write('test_trajectory_scalars.py [track file] ' \
        '[texture] [out file] [first step] [last step]\n')
    sys.exit(-1)

tracks_file = sys.argv[1]
tex = sys.argv[2]
out_file = sys.argv[3]
start_index = int(sys.argv[4])
end_index = int(sys.argv[5])

# construct the pipeline
reader = teca_table_reader.New()
reader.set_file_name(tracks_file)
reader.set_index_column('track_id')
reader.set_generate_original_ids(1)

calendar = teca_table_calendar.New()
calendar.set_input_connection(reader.get_output_port())
calendar.set_time_column('time')

scalars = teca_tc_trajectory_scalars.New()
scalars.set_input_connection(calendar.get_output_port())
scalars.set_texture(tex)
scalars.set_basename('test_trajectory_scalars')

mapper = teca_table_reduce.New()
mapper.set_input_connection(scalars.get_output_port())
mapper.set_start_index(start_index)
mapper.set_end_index(end_index)
mapper.set_verbose(1)
mapper.set_thread_pool_size(1)

sort = teca_table_sort.New()
sort.set_input_connection(mapper.get_output_port())
sort.set_index_column('original_ids')

if os.path.exists(out_file):
  baseline = teca_table_reader.New()
  baseline.set_file_name(out_file)

  diff = teca_dataset_diff.New()
  diff.set_input_connection(0, baseline.get_output_port())
  diff.set_input_connection(1, sort.get_output_port())
  diff.update()

else:
  #write data
  writer = teca_table_writer.New()
  writer.set_input_connection(sort.get_output_port())
  writer.set_file_name(out_file)
  writer.update()
