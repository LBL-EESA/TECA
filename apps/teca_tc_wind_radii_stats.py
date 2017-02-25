#!/usr/bin/env python
import sys
from teca import *
import argparse
import matplotlib
import numpy as np

# parse the command line
parser = argparse.ArgumentParser()
parser.add_argument('track_file', type=str,
    help='file containing TC storm tracks')

parser.add_argument('output_prefix', type=str,
    help="prefix to output files")

parser.add_argument('-d', '--dpi', type=int,
    default=100, help="output image DPI")

parser.add_argument('-i', '--interact', action='store_true',
    help="display plots in pop-up windows")

parser.add_argument('--wind_column', type=str,
    default='surface_wind', help="output image DPI")

args = parser.parse_args()

# configure matplotlib
if not args.interact:
    matplotlib.use('Agg')

# load the track table
tr = teca_table_reader.New()
tr.set_file_name(args.track_file)

wr = teca_tc_wind_radii_stats.New()
wr.set_input_connection(tr.get_output_port())
wr.set_interactive(args.interact)
wr.set_dpi(args.dpi)
wr.set_output_prefix(args.output_prefix)
wr.set_wind_column(args.wind_column)

wr.update()
