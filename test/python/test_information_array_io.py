import sys, os
import numpy as np
from teca import *

set_stack_trace_on_error()

if len(sys.argv) != 4:
    sys.stderr.write('test_information_array_io.py [n steps] ' \
        '[steps per file] [baseline file]\n')
    sys.exit(-1)

n_steps = int(sys.argv[1])
steps_per_file = int(sys.argv[2])
baseline = sys.argv[3]

out_file = 'test_information_arrays-%t%.nc'
files_regex = 'test_information_arrays.*\\.nc$'
vrb = 1

class generate_info_arrays(teca_python_algorithm):

    def __init__(self):
        self.verbose = 0

    def set_verbose(self, val):
        self.verbose = val

    def print_status(self, msg):
        if self.verbose:
            sys.stderr.write('generate_info_arrays::%s\n'%(msg))

    def get_report_callback(self):
        self.print_status('get_report_callback')

        def report_callback(port, md_in):
            self.print_status('report_callback')
            md_out = teca_metadata(md_in[0])
            try:
                arrays = md_out['arrays']
            except:
                arrays = []
            md_out['arrays'] = arrays + ['time_info', 'step_info']
            return md_out

        return report_callback

    def get_execute_callback(self):
        self.print_status('get_execute_callback')

        def execute_callback(port, data_in, req_in):
            self.print_status('execute_callback')

            mesh_in = as_const_teca_cartesian_mesh(data_in[0])
            t = mesh_in.get_time()
            s = mesh_in.get_time_step()

            mesh_out = teca_cartesian_mesh.New()
            mesh_out.shallow_copy(mesh_in)

            t_inf = teca_variant_array.New(np.array([t, t, t, t]))
            mesh_out.get_information_arrays().append('time_info', t_inf)

            s_inf = teca_variant_array.New(np.array([s,s]))
            mesh_out.get_information_arrays().append('step_info', s_inf)

            if self.verbose:
                sys.stderr.write('t=%g, time_info=%s\n'%(t, str(t_inf)))
                sys.stderr.write('s=%d, step_info=%s\n'%(s, str(s_inf)))

            return mesh_out

        return execute_callback


class print_info_arrays(teca_python_algorithm):

    def __init__(self):
        self.verbose = 0

    def set_verbose(self, val):
        self.verbose = val

    def print_status(self, msg):
        if self.verbose:
            sys.stderr.write('print_info_arrays::%s\n'%(msg))

    def get_execute_callback(self):
        self.print_status('get_execute_callback')

        def execute_callback(port, data_in, req_in):
            self.print_status('execute_callback')

            mesh_in = as_const_teca_cartesian_mesh(data_in[0])
            mesh_out = teca_cartesian_mesh.New()
            mesh_out.shallow_copy(mesh_in)

            if self.verbose:
                sys.stderr.write('t=%g, time_info=%s\n'%(mesh_out.get_time(), \
                    str(mesh_out.get_information_arrays().get('time_info'))))

                sys.stderr.write('s=%d, step_info=%s\n'%(mesh_out.get_time_step(), \
                    str(mesh_out.get_information_arrays().get('step_info'))))

            return mesh_out

        return execute_callback





# construct a small mesh with 8 time values
src = teca_cartesian_mesh_source.New()
src.set_whole_extents([0, 1, 0, 1, 0, 1, 0, n_steps-1])
src.set_bounds([0.0, 360.0, -90.0, 90.0, 0.0, 0.0, 1.0, 8.0])
src.set_calendar('standard')
src.set_time_units('days since 2019-09-24')

# add some information arrays
gia = generate_info_arrays.New()
gia.set_input_connection(src.get_output_port())
gia.set_verbose(vrb)

# write the data
wex = teca_index_executive.New()
wex.set_verbose(vrb)

cfw = teca_cf_writer.New()
cfw.set_input_connection(gia.get_output_port())
cfw.set_file_name(out_file)
cfw.set_steps_per_file(steps_per_file)
cfw.set_thread_pool_size(1)
cfw.set_executive(wex)
cfw.update()

# read the data back in
cfr = teca_cf_reader.New()
cfr.set_files_regex(files_regex)
md = cfr.update_metadata()

# print it out
par = print_info_arrays.New()
par.set_input_connection(cfr.get_output_port())
par.set_verbose(vrb)

rex = teca_index_executive.New()
rex.set_arrays(['time_info', 'step_info'])
rex.set_verbose(vrb)

if os.path.exists(baseline):
    sys.stderr.write('running the test...\n')

    cmr = teca_cartesian_mesh_reader.New()
    cmr.set_file_name(baseline)

    diff = teca_dataset_diff.New()
    diff.set_input_connection(0, cmr.get_output_port())
    diff.set_input_connection(1, par.get_output_port())
    diff.set_executive(rex)
    diff.update()
else:
    sys.stderr.write('writing the baseline...\n')

    cmw = teca_cartesian_mesh_writer.New()
    cmw.set_file_name(baseline)
    cmw.set_input_connection(par.get_output_port())
    cmw.set_file_name(baseline)
    cmw.set_executive(rex)
    cmw.update()
