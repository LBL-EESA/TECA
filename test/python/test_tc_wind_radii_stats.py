#!/usr/bin/env python
import sys
import os
from teca import *

if len(sys.argv) != 2:
    sys.stderr.write('ERROR:\ntest_tc_wind_radii_stats.py [input]\n\n')
    sys.exit(-1)

infile = sys.argv[1]

reader = teca_table_reader.New()
reader.set_file_name(infile)

stats = teca_tc_wind_radii_stats.New()
stats.set_input_connection(reader.get_output_port())
stats.update()
