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

# compute the point-wise average of two variables v0 and v1.
# for every i
#
# avg[i] = (v0[i] + v1[i])/2
#
def get_point_average_execute(rank, v0_name, v1_name, res_name):
    def execute(port, data_in, req):
        sys.stderr.write('point_average::execute MPI %d\n'%(rank))

        in_mesh = as_teca_cartesian_mesh(data_in[0])

        v0 = in_mesh.get_point_arrays().get(v0_name).as_array()
        v1 = in_mesh.get_point_arrays().get(v1_name).as_array()

        res = (v0 + v1)/2.0

        out_mesh = teca_cartesian_mesh.New()
        out_mesh.shallow_copy(in_mesh)
        out_mesh.get_point_arrays().append(res_name, res)

        return out_mesh
    return execute

# compute the point-wise difference of two variables v0 and v1.
# for every i
#
# diff[i] = v1[i] - v0[i]
#
def get_point_difference_execute(rank, v0_name, v1_name, res_name):
    def execute(port, data_in, req):
        sys.stderr.write('point_difference::execute MPI %d\n'%(rank))

        in_mesh = as_teca_cartesian_mesh(data_in[0])

        v0 = in_mesh.get_point_arrays().get(v0_name).as_array()
        v1 = in_mesh.get_point_arrays().get(v1_name).as_array()

        res = v1 - v0

        out_mesh = teca_cartesian_mesh.New()
        out_mesh.shallow_copy(in_mesh)
        out_mesh.get_point_arrays().append(res_name, res)

        return out_mesh
    return execute

if (len(sys.argv) != 18):
    sys.stderr.write('\n\nUsage error:\n' \
        'test_tc_candidates [input regex] [output] [first step] [last step] [n threads] ' \
        '[n_omp_threads] [850 mb wind x] [850 mb wind y] [surface wind x] [surface wind y] ' \
        '[surface pressure] [500 mb temp] [200 mb temp] [1000 mb z] [200 mb z] [low lat] ' \
        '[high lat]\n\n')
    sys.exit(-1)

# parse command line
regex = sys.argv[1]
baseline = sys.argv[2]
start_index = int(sys.argv[3])
end_index = int(sys.argv[4])
n_threads = int(sys.argv[5])
n_omp_threads = int(sys.argv[6])
ux_850mb = sys.argv[7]
uy_850mb = sys.argv[8]
ux_surf = sys.argv[9]
uy_surf = sys.argv[10]
P_surf = sys.argv[11]
T_500mb = sys.argv[12]
T_200mb = sys.argv[13]
z_1000mb = sys.argv[14]
z_200mb = sys.argv[15]
low_lat = float(sys.argv[16])
high_lat = float(sys.argv[17])

if (rank == 0):
    sys.stderr.write('Testing on %d MPI processes %d threads %d omp threads\n'%(n_ranks, n_threads, n_omp_threads))


# create the pipeline objects
cf_reader = teca_cf_reader.New()
cf_reader.set_files_regex(regex)

coords = teca_normalize_coordinates.New()
coords.set_input_connection(cf_reader.get_output_port())

# surface wind speed
surf_wind = teca_l2_norm.New()
surf_wind.set_input_connection(coords.get_output_port())
surf_wind.set_component_0_variable(ux_surf)
surf_wind.set_component_1_variable(uy_surf)
surf_wind.set_l2_norm_variable('surface_wind')

# vorticity at 850mb
vort_850mb = teca_vorticity.New()
vort_850mb.set_input_connection(surf_wind.get_output_port())
vort_850mb.set_component_0_variable(ux_850mb)
vort_850mb.set_component_1_variable(uy_850mb)
vort_850mb.set_vorticity_variable('850mb_vorticity')

# core temperature
core_temp = teca_derived_quantity.New()
core_temp.set_input_connection(vort_850mb.get_output_port())
core_temp.set_dependent_variables([T_500mb, T_200mb])
core_temp.set_derived_variable('core_temperature')
core_temp.set_execute_callback(
    get_point_average_execute(rank, T_500mb, T_200mb, 'core_temperature'))

# thickness
thickness = teca_derived_quantity.New()
thickness.set_input_connection(core_temp.get_output_port())
thickness.set_dependent_variables([z_1000mb, z_200mb])
thickness.set_derived_variable('thickness')
thickness.set_execute_callback(
    get_point_difference_execute(rank, z_1000mb, z_200mb, 'thickness'))

# candidate detection
cand = teca_tc_candidates.New()
cand.set_input_connection(thickness.get_output_port())
cand.set_surface_wind_speed_variable('surface_wind')
cand.set_vorticity_850mb_variable('850mb_vorticity')
cand.set_sea_level_pressure_variable(P_surf)
cand.set_core_temperature_variable('core_temperature')
cand.set_thickness_variable('thickness')
cand.set_max_core_radius(2.0)
cand.set_min_vorticity_850mb(1.6e-4)
cand.set_vorticity_850mb_window(7.74446)
cand.set_max_pressure_delta(400.0)
cand.set_max_pressure_radius(5.0)
cand.set_max_core_temperature_delta(0.8)
cand.set_max_core_temperature_radius(5.0)
cand.set_max_thickness_delta(50.0)
cand.set_max_thickness_radius(4.0)
cand.set_search_lat_low(low_lat)
cand.set_search_lat_high(high_lat)
#cand.set_search_lon_low()
#cand.set_search_lon_high()
cand.set_omp_num_threads(n_omp_threads)

# map-reduce
map_reduce = teca_table_reduce.New()
map_reduce.set_input_connection(cand.get_output_port())
map_reduce.set_start_index(start_index)
map_reduce.set_end_index(end_index)
map_reduce.set_verbose(1)
map_reduce.set_thread_pool_size(n_threads)

# sort results by wind speed, this is gives the test output a deterministic
# order independent of how many OpenMP threads are used
sort = teca_table_sort.New()
sort.set_input_connection(map_reduce.get_output_port())
sort.set_index_column('surface_wind')

# compute dates
cal = teca_table_calendar.New()
cal.set_input_connection(sort.get_output_port())

do_test = system_util.get_environment_variable_bool('TECA_DO_TEST', True)
if do_test and os.path.exists(baseline):
  # run the test
  sys.stderr.write('running the test ... \n')

  table_reader = teca_table_reader.New()
  table_reader.set_file_name(baseline)

  diff = teca_dataset_diff.New()
  diff.set_input_connection(0, table_reader.get_output_port())
  diff.set_input_connection(1, cal.get_output_port())
  diff.set_verbose(1)

 # depends on the number of OpenMP threads
  diff.set_skip_array('storm_id')

  diff.update()

else:
  # write the data
  sys.stderr.write('generating the baseling "%s"\n'%(baseline))

  table_writer = teca_table_writer.New()
  table_writer.set_input_connection(cal.get_output_port())
  table_writer.set_file_name(baseline)
  #table_writer.set_output_format_csv()
  table_writer.set_output_format_bin()

  table_writer.update();
