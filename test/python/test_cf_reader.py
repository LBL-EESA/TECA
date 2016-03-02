from teca_py_core import *
from teca_py_io import *
from teca_py_alg import *
import sys

argc = len(sys.argv)
if not argc >= 5:
    sys.stderr.write('test_cf_reader.py [dataset regex] ' \
        '[first step] [last step] [out file name] ' \
        '[array 1] ... [array n]\n')
    sys.exit(-1)

data_regex = sys.argv[1]
first_step = int(sys.argv[2])
last_step = int(sys.argv[3])
out_file = sys.argv[4]
arrays = []
i = 5
while i < argc:
    arrays.append(sys.argv[i])
    i += 1

cfr = teca_cf_reader.New()
cfr.set_files_regex(data_regex)
cfr.set_x_axis_variable('lon')
cfr.set_y_axis_variable('lat')
cfr.set_t_axis_variable('time')

exe = teca_time_step_executive.New()
exe.set_first_step(first_step)
exe.set_last_step(last_step)
exe.set_arrays(arrays)

wri = teca_vtk_cartesian_mesh_writer.New()
wri.set_input_connection(cfr.get_output_port())
wri.set_executive(exe)
wri.set_file_name(out_file)

wri.update()
