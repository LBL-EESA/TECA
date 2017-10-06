#!/usr/bin/env python
from mpi4py import *
from teca import *
import sys
import argparse

# parse the command line
parser = argparse.ArgumentParser()
parser.add_argument('files_regex', type=str,
    help='regex describing NetCDF CF2 input to read')

parser.add_argument('out_file', type=str,
    help='path to write result')

parser.add_argument('--select', type=str, required=False,
    help='a logical expression on table columns. '      \
         'Row where this evaluates to true are passed ' \
         'to the output')

args = parser.parse_args()

# CF2 reader
reader = teca_cf_reader.New()
reader.set_files_regex(args.files_regex)
reader.set_x_axis_variable('lon')
reader.set_y_axis_variable('lat')
reader.set_z_axis_variable('lev')
reader.set_t_axis_variable('time')

# add IVT here
ivt = teca_ivt.New()
ivt.set_input_connection(reader.get_output_port())
#TODO: set these using command line arguments
uwind_var = 'u'
vwind_var = 'v'
qv_var = 'QV'
ivt.set_array_dict(zonal_wind = uwind_var, \
                   meridional_wind = vwind_var, \
                   water_vapor = qv_var)
# flag to use lev as pressure
# TODO: add a command line flag for this
ivt.use_z_coord_as_pressure(True)
ivt.set_bounds([-180.0, 180.0, -90.0, 90.0, 0., 1050.])

# add AR detector here
detector = teca_ar_rutz.New()
detector.set_input_connection(ivt.get_output_port())
detector.set_bounds([-180.0, 180.0, -90.0, 90.0, 950.0, 950.0])

# map reduce
mapper = teca_table_reduce.New()
mapper.set_input_connection(detector.get_output_port())
mapper.set_thread_pool_size(1)

# write the table back out
writer = teca_table_writer.New()
writer.set_input_connection(mapper.get_output_port())
writer.set_file_name(args.out_file)

# execute the pipeline
writer.update()

