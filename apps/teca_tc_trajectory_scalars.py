#!/usr/bin/env python
try:
    from mpi4py import *
    rank = MPI.COMM_WORLD.Get_rank()
    n_ranks = MPI.COMM_WORLD.Get_size()
except:
    rank = 0
    n_ranks = 1
import sys
from teca import *
import argparse
import matplotlib

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

parser.add_argument('--first_track', type=int,
    default=0, help='first track id to process (0)')

parser.add_argument('--last_track', type=int,
    default=-1, help='last track id to process (-1)')

parser.add_argument('--texture', type=str,
    default='', help='path to background image for track plots')

parser.add_argument('--axes_scaled', action='store_false',
    help="distort aspect ratio in geographic plots to fit window")

parser.add_argument('--plot_peak_radius', action='store_true',
    help="include the peak radius in the plots")

args = parser.parse_args()

# configure matplotlib
if not args.interact:
    matplotlib.use('Agg')

# construct the pipeline
reader = teca_table_reader.New()
reader.set_file_name(args.tracks_file)
reader.set_index_column('track_id')

calendar = teca_table_calendar.New()
calendar.set_input_connection(reader.get_output_port())
calendar.set_time_column('time')

scalars = teca_tc_trajectory_scalars.New()
scalars.set_input_connection(calendar.get_output_port())
scalars.set_basename(args.output_prefix)
scalars.set_texture(args.texture)
scalars.set_dpi(args.dpi)
scalars.set_interactive(args.interact)
scalars.set_axes_equal(args.axes_scaled)
scalars.set_plot_peak_radius(args.plot_peak_radius)

mapper = teca_table_reduce.New()
mapper.set_input_connection(scalars.get_output_port())
mapper.set_first_step(args.first_track)
mapper.set_last_step(args.last_track)
mapper.set_thread_pool_size(1)

# execute
mapper.update()
