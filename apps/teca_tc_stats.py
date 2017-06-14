#!/usr/bin/env python
import sys
from teca import *
import argparse

# parse the command line
parser = argparse.ArgumentParser()
parser.add_argument('tracks_file', type=str,
    help='file containing TC storm tracks')

parser.add_argument('output_prefix', type=str,
    help="prefix to output files")

parser.add_argument('-d', '--dpi', type=int,
    default=100, help="output image DPI")

parser.add_argument('-i', '--interact', action='store_true',
    help="display plots in pop-up windows")

parser.add_argument('-a', '--ind_axes', action='store_false',
    help="normalize y-axis in grouped plots")

args = parser.parse_args()

# construct the pipeline
reader = teca_table_reader.New()
reader.set_file_name(args.tracks_file)

classify = teca_tc_classify.New()
classify.set_input_connection(reader.get_output_port())

calendar = teca_table_calendar.New()
calendar.set_input_connection(classify.get_output_port())
calendar.set_time_column('start_time')

writer = teca_table_writer.New()
writer.set_input_connection(calendar.get_output_port())
writer.set_file_name('%s_class_table.csv'%(args.output_prefix))

act = teca_tc_activity.New()
act.set_input_connection(writer.get_output_port())
act.set_basename(args.output_prefix)
act.set_dpi(args.dpi)
act.set_interactive(args.interact)
act.set_rel_axes(args.ind_axes)

stats = teca_tc_stats.New()
stats.set_input_connection(act.get_output_port())
stats.set_basename(args.output_prefix)
stats.set_dpi(args.dpi)
stats.set_interactive(args.interact)
stats.set_rel_axes(args.ind_axes)

# execute
act.update()
