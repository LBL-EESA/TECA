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

if len(sys.argv) < 7:
    sys.stderr.write('teca_jet_detect [dataset regex] ' \
        '[out file name] [first step] [last step] [n threads]' \
        '[wind u variable] [wind v variable]\n')
    sys.exit(-1)

data_regex = sys.argv[1]
out_file = sys.argv[2]
first_step = int(sys.argv[3])
last_step = int(sys.argv[4])
n_threads = int(sys.argv[5])
wind_u_var = sys.argv[6]
wind_v_var = sys.argv[7]

# TODO -- specify in terms of lat lon
extent = [192, 512, 384, 682, 9, 9]

def get_request(wind_speed_var, extent):
    """
    returns the function that implements the request
    phase of the TECA pipeline.

    wind_speed_var_name - name of variable containing wind speed
    extent - describes the subset of the data to load
    """
    def request(port, md_in, req_in):
        req = teca_metadata(req_in)
        req['arrays'] = [wind_speed_var]
        req['extent'] = extent
        return [req]
    return request

def get_execute(wind_speed_var):
    """
    returns the function that implements the execute
    phase of the TECA pipeline.

    wind_speed_var_name - name of variable containing wind speed
    """
    def execute(port, data_in, req):
        # get the input as a mesh
        mesh = as_teca_cartesian_mesh(data_in[0])

        # get metadata
        step = mesh.get_time_step()
        time = mesh.get_time()

        extent = mesh.get_extent()

        # get the dimensions of the data
        nlon = extent[1] - extent[0] + 1
        nlat = extent[3] - extent[2] + 1

        # get the coordinate arrays
        lon = mesh.get_x_coordinates().as_array()
        lat = mesh.get_y_coordinates().as_array()

        # get the wind speed values as an numpy array
        wind = mesh.get_point_arrays().get(wind_speed_var).as_array()
        wind = wind.reshape([nlat, nlon])

        # for each lon find lat where max wind occurs
        lat_ids = np.argmax(wind, axis=0)
        avg_lat = np.average(lat[lat_ids])
        max_val = np.max(wind[lat_ids, np.arange(nlon)])

        # put it into a table
        table = teca_table.New()
        table.copy_metadata(mesh)

        table.declare_columns(['step','time','avg_lat', \
            'max_wind_speed'], ['ul','d','d','d'])

        table << mesh.get_time_step() << mesh.get_time() \
            << float(avg_lat) << float(max_val)

        return table
    return execute

if (rank == 0):
    sys.stderr.write('Testing on %d MPI processes %d threads\n'%(n_ranks, n_threads))

cfr = teca_cf_reader.New()
cfr.set_files_regex(data_regex)
cfr.set_x_axis_variable('lon')
cfr.set_y_axis_variable('lat')
cfr.set_z_axis_variable('plev')
cfr.set_t_axis_variable('time')

l2 = teca_l2_norm.New()
l2.set_input_connection(cfr.get_output_port())
l2.set_component_0_variable(wind_u_var)
l2.set_component_1_variable(wind_v_var)
l2.set_l2_norm_variable('wind_speed')

jet_detect = teca_programmable_algorithm.New()
jet_detect.set_request_callback(get_request('wind_speed', extent))
jet_detect.set_execute_callback(get_execute('wind_speed'))
jet_detect.set_input_connection(l2.get_output_port())

mr = teca_table_reduce.New()
mr.set_input_connection(jet_detect.get_output_port())
mr.set_first_step(first_step)
mr.set_last_step(last_step)
mr.set_thread_pool_size(n_threads)

sort = teca_table_sort.New()
sort.set_input_connection(mr.get_output_port())
sort.set_index_column('time')

cal = teca_table_calendar.New()
cal.set_input_connection(sort.get_output_port())

tw = teca_table_writer.New()
tw.set_input_connection(cal.get_output_port())
tw.set_file_name(out_file)

tw.update()
