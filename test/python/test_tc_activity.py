#!/usr/bin/env python
import sys
import os
from teca import *

if len(sys.argv) != 3:
    sys.stderr.write('ERROR:\ntest_tc_activity.py [input] [baseline]\n\n')
    sys.exit(-1)

infile = sys.argv[1]
baseline = sys.argv[2]

reader = teca_table_reader.New()
reader.set_file_name(infile)

cal = teca_table_calendar.New()
cal.set_input_connection(reader.get_output_port())
cal.set_time_column('start_time')

activity = teca_tc_activity.New()
activity.set_input_connection(cal.get_output_port())

if os.path.exists(baseline):
    table_reader = teca_table_reader.New()
    table_reader.set_file_name(baseline)
    diff = teca_dataset_diff.New()
    diff.set_input_connection(0, table_reader.get_output_port())
    diff.set_input_connection(1, activity.get_output_port())
    diff.update()
else:
    sys.stderr.write('generating baseline\n')
    table_writer = teca_table_writer.New()
    table_writer.set_input_connection(activity.get_output_port())
    table_writer.set_file_name(baseline)
    table_writer.update();
