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
reader.set_t_axis_variable('')

# add AR detector here
detector = teca_ar_sellers.New()
detector.set_input_connection(reader.get_output_port())
detector.set_arrays(['u','v','QV'])
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
