from teca import *
import numpy as np
import sys

set_stack_trace_on_error()

if not len(sys.argv) == 7:
    sys.stderr.write('test_python_algorithm.py [dataset regex] ' \
        '[u_var] [v_var] [first step] [last step] [out file name]\n')
    sys.exit(-1)

data_regex = sys.argv[1]
u_var = sys.argv[2]
v_var = sys.argv[3]
first_step = int(sys.argv[4])
end_index = int(sys.argv[5])
out_file = sys.argv[6]

class wind_speed(teca_python_algorithm):

    def __init__(self):
        self.u_var = None
        self.v_var = None

    def set_u_var(self, u_var):
        self.u_var = u_var

    def set_v_var(self, v_var):
        self.v_var = v_var

    def report(self, port, md_in):
        sys.stderr.write('wind_speed::report\n')
        md = md_in[0]
        md.append('variables', 'wind_speed')
        return md

    def request(self, port, md_in, req_in):
        sys.stderr.write('wind_speed::request\n')
        req = teca_metadata(req_in)
        req['arrays'] = [self.u_var, self.v_var]
        return [req]

    def execute(self, port, data_in, req):
        sys.stderr.write('wind_speed::execute\n')
        in_mesh = as_teca_cartesian_mesh(data_in[0])
        out_mesh = teca_cartesian_mesh.New()
        out_mesh.shallow_copy(in_mesh)
        arrays = out_mesh.get_point_arrays()
        u = arrays[self.u_var]
        v = arrays[self.v_var]
        w = np.sqrt(u*u + v*v)
        arrays['wind_speed'] = w
        return out_mesh



class max_wind_speed(wind_speed):

    def execute(self, port, data_in, req):
        """
        override the base class implementation.
        demonstrate the use of super to call the base
        class
        """
        sys.stderr.write('max_wind_speed::execute\n')

        # let the base class calculate the wind speed
        wdata = super().execute(port, data_in, req)

        # find the max
        mesh = as_teca_cartesian_mesh(wdata)
        md = mesh.get_metadata()
        arrays = mesh.get_point_arrays()
        max_ws = np.max(arrays['wind_speed'])

        # construct the output
        table = teca_table.New()
        table.copy_metadata(mesh)

        table.declare_columns(['time_step', 'time', 'max_wind'], \
            ['l', 'd', 'd'])

        table << md['time_step'] << md['time'] << max_ws

        return table



cfr = teca_cf_reader.New()
cfr.set_files_regex(data_regex)
cfr.set_x_axis_variable('lon')
cfr.set_y_axis_variable('lat')
cfr.set_t_axis_variable('time')

coords = teca_normalize_coordinates.New()
coords.set_input_connection(cfr.get_output_port())

mws = max_wind_speed.New()
mws.set_input_connection(coords.get_output_port())
mws.set_u_var(u_var)
mws.set_v_var(v_var)

mr = teca_table_reduce.New()
mr.set_input_connection(mws.get_output_port())
mr.set_thread_pool_size(1)

ts = teca_table_sort.New()
ts.set_input_connection(mr.get_output_port())
ts.set_index_column('time_step')

tw = teca_table_writer.New()
tw.set_input_connection(ts.get_output_port())
tw.set_file_name('test_python_alg.csv')

tw.update()

