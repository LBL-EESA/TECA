from teca import *
import numpy
if get_teca_has_cupy():
    import cupy
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
        dev = -1
        np = numpy
        if get_teca_has_cuda() and get_teca_has_cupy():
            dev = req['device_id']
            if dev >= 0:
                cupy.cuda.Device(dev).use()
                np = cupy
        dev_str = 'CPU' if dev < 0 else 'GPU %d'%(dev)
        sys.stderr.write('wind_speed::execute %s\n'%(dev_str))
        in_mesh = as_teca_mesh(data_in[0])
        out_mesh = as_teca_mesh(in_mesh.new_instance())
        out_mesh.shallow_copy(in_mesh)
        arrays = out_mesh.get_point_arrays()
        if dev < 0:
            u = arrays[self.u_var].get_host_accessible()
            v = arrays[self.v_var].get_host_accessible()
        else:
            u = arrays[self.u_var].get_cuda_accessible()
            v = arrays[self.v_var].get_cuda_accessible()
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
        # get the device to run on
        dev = -1
        np = numpy
        if get_teca_has_cuda() and get_teca_has_cupy():
            dev = req['device_id']
            if dev >= 0:
                cupy.cuda.Device(dev).use()
                np = cupy

        # report
        dev_str = 'CPU' if dev < 0 else 'GPU %d'%(dev)
        sys.stderr.write('max_wind_speed::execute %s\n'%(dev_str))

        # let the base class calculate the wind speed
        wdata = super().execute(port, data_in, req)

        # find the max
        mesh = as_teca_mesh(wdata)
        arrays = mesh.get_point_arrays()
        if dev < 0:
            ws = arrays['wind_speed'].get_host_accessible()
        else:
            ws = arrays['wind_speed'].get_cuda_accessible()
        max_ws = np.max(ws)

        # construct the output
        table = teca_table.New()
        table.copy_metadata(mesh)

        table.declare_columns(['time_step', 'time', 'max_wind'], \
            ['l', 'd', 'd'])

        table << mesh.get_time_step() << mesh.get_time() << max_ws

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
