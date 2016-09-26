#!/usr/bin/env python
from teca import *
import sys
import argparse

# parse the command line
parser = argparse.ArgumentParser()
parser.add_argument('in_file', type=str,
    help='file containing events')

parser.add_argument('out_file', type=str,
    help='file to write filtred events to')

parser.add_argument('--time_column', type=str, required=False,
    help='name of column containing time axis')

parser.add_argument('--start_time', type=float, required=False,
    default=float('-inf'), help='filter out events occuring before this time')

parser.add_argument('--end_time', type=float, required=False,
    default=float('+inf'), help='filter out events occuring after this time')

parser.add_argument('--step_column', type=str, required=False,
    help='name of column containing time steps')

parser.add_argument('--step_interval', type=int, required=False,
    default=1, help='filter out time steps modulo this interval')

parser.add_argument('--x_coordinate_column', type=str, required=False,
    help='name of columnevent containing event x coordinates')

parser.add_argument('--y_coordinate_column', type=str,
    required=False, help='name of column containing event y coordinates')

parser.add_argument('--region_x_coords', nargs='+', type=float,
    required=False, help='x coordinates defining region to filter')

parser.add_argument('--region_y_coords', nargs='+', type=float,
    required=False, help='y coordinates defining region to filter')

parser.add_argument('--region_sizes', nargs='+', type=int,
    required=False, help='sizes of each of the regions')

args = parser.parse_args()

# build and configure the pipeline
r = teca_table_reader.New()
r.set_file_name(args.in_file)

f = teca_event_filter.New()
f.set_input_connection(r.get_output_port())

if (args.time_column):
    f.set_time_column(args.time_column)
    f.set_start_time(args.start_time)
    f.set_end_time(args.end_time)

if (args.step_column):
    f.set_step_column(args.step_column)
    f.set_step_interval(args.step_interval)

if (args.x_coordinate_column):
    try:
        f.set_x_coordinate_column(args.x_coordinate_column)
        f.set_y_coordinate_column(args.y_coordinate_column)
        f.set_region_x_coordinates(args.region_x_coords)
        f.set_region_y_coordinates(args.region_y_coords)
        f.set_region_sizes(args.region_sizes)
    except Exception as e:
        sys.stderr.write('ERROR: %s\nFor spatial filtering you must specify ' \
            ' all of:\n  x_coordinate_column\n  y_coordinate_column\n' \
            '  region_x_coords\n  region_y_coords\n  region_size\n'%(str(e)))
        sys.exit(-1)

w = teca_table_writer.New()
w.set_input_connection(f.get_output_port())
w.set_file_name(args.out_file)

# run it
w.update()
