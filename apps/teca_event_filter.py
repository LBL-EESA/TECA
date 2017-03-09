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
tr = teca_table_reader.New()
tr.set_file_name(args.in_file)

# tip of the pipeline is held in a temp
ptip = tr
expr = ''

if (args.time_column):
    ee = teca_evaluate_expression.New()
    ee.set_input_connection(ptip.get_output_port())
    ee.set_expression('(%s >= %g) && (%s <= %g)'%( \
        args.time_column, args.start_time, \
        args.time_column, args.end_time))
    ee.set_result_variable('in_time')
    ptip = ee
    expr = 'in_time'

if (args.step_column):
    ee = teca_evaluate_expression.New()
    ee.set_input_connection(ptip.get_output_port())
    ee.set_expression('%s %% %d'%( \
        args.step_column, args.step_interval))
    ptip = ee
    expr = expr + ' && in_step' if expr else 'in_step'

if (args.x_coordinate_column):
    try:
        rm = teca_table_region_mask.New()
        rm.set_input_connection(ptip.get_output_port())
        rm.set_x_coordinate_column(args.x_coordinate_column)
        rm.set_y_coordinate_column(args.y_coordinate_column)
        rm.set_region_x_coordinates(args.region_x_coords)
        rm.set_region_y_coordinates(args.region_y_coords)
        rm.set_region_sizes(args.region_sizes)
        rm.set_result_column('in_region')
        ptip = rm
        expr = expr + ' && in_region' if expr else 'in_region'
    except Exception as e:
        sys.stderr.write('ERROR: %s\nFor spatial filtering you must specify' \
            ' all of:\n  x_coordinate_column\n  y_coordinate_column\n' \
            '  region_x_coords\n  region_y_coords\n  region_size\n'%(str(e)))
        sys.exit(-1)

if not expr:
    sys.stderr.write('ERROR: must specify one of:\n' \
        '  --time_column\n  --step_column\n  --x_coordinate_column\n')
    sys.exit(-1)

rr = teca_table_remove_rows.New()
rr.set_input_connection(ptip.get_output_port())
rr.set_remove_dependent_variables(1)
rr.set_mask_expression('!(%s)'%(expr))

tw = teca_table_writer.New()
tw.set_input_connection(rr.get_output_port())
tw.set_file_name(args.out_file)

# run it
tw.update()
