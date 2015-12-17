# initialize MPI
from mpi4py import MPI
# bring in TECA
from teca_py_io import *
from teca_py_alg import *

# start the pipeline with the NetCDF CF-2.0 reader
cfr = teca_cf_reader.New()
cfr.set_files_regex('cam5_1_amip_run2\.cam2\.h2\.*')
cfr.set_x_axis_variable('lon')
cfr.set_y_axis_variable('lat')
cfr.set_t_axis_variable('time')

# add L2 norm operator to compute wind speed
l2n = teca_l2_norm.New()
l2n.set_component_0_variable('U850')
l2n.set_component_1_variable('V850')
l2n.set_l2_norm_variable('wind_speed')
l2n.set_input_connection(cfr.get_output_port())

# and vorticity operator to compute wind vorticity
vor = teca_vorticity.New()
vor.set_component_0_variable('U850')
vor.set_component_1_variable('V850')
vor.set_vorticity_variable('wind_vorticity')
vor.set_input_connnection(l2n.get_output_port())

# and finally the tropical cyclone detector
tcd = teca_tc_detect.New()
tcd.set_pressure_variable('PSL')
tcd.set_temperature_variable('TMQ')
tcd.set_wind_speed_variable('wind_speed')
tcd.set_vorticity_variable('wind_vorticity')
tcd.set_input_connection(vor.get_output_port())

# now add the map-reduce, the pipeline above is run in
# parallel using MPI+threads. Each thread processes one time
# step. the pipeline below this algorithm runs in serial on
# rank 0, # with 1 thread
mapr = teca_table_reduce.New()
mapr.set_thread_pool_size(2)
mapr.set_first_step(0)
mapr.set_last_step(-1)
mapr.set_input_connection(tcd.get_output_port())

# save the detected stroms
twr = teca_table_writer.New()
twr.set_file_name('detections_%t%.csv')
twr.set_input_connection(mapr.get_output_port())

# the commands above connect and configure the pipeline
# this command actually runs it
twr.update()
