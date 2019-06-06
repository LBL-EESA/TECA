try:
    from mpi4py import *
    rank = MPI.COMM_WORLD.Get_rank()
    n_ranks = MPI.COMM_WORLD.Get_size()
except:
    rank = 0
    n_ranks = 1
from teca import *
import sys

set_stack_trace_on_error()

argc = len(sys.argv)
if not argc >= 7:
    sys.stderr.write('test_cf_reader.py [dataset regex] ' \
        '[first step] [last step] [n threads] [steps per file] ' \
        '[out file name] [array 1] ... [array n]\n')
    sys.exit(-1)

data_regex = sys.argv[1]
first_step = int(sys.argv[2])
end_index = int(sys.argv[3])
n_threads = int(sys.argv[4])
steps_per_file = int(sys.argv[5])
out_file = sys.argv[6]
arrays = []
i = 7
while i < argc:
    arrays.append(sys.argv[i])
    i += 1

cfr = teca_cf_reader.New()
cfr.set_files_regex(data_regex)
cfr.set_x_axis_variable('lon')
cfr.set_y_axis_variable('lat')
cfr.set_t_axis_variable('time')

coords = teca_normalize_coordinates.New()
coords.set_input_connection(cfr.get_output_port())

exe = teca_index_executive.New()
exe.set_start_index(first_step)
exe.set_end_index(end_index)
exe.set_arrays(arrays)

wri = teca_cf_writer.New()
wri.set_input_connection(coords.get_output_port())
wri.set_thread_pool_size(n_threads)
wri.set_steps_per_file(steps_per_file)
wri.set_executive(exe)
wri.set_file_name(out_file)

wri.update()
