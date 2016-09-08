#!/usr/bin/env python
from teca import *
import sys

if (len(sys.argv) != 3):
    sys.stderr.write('usage:\nteca_convert_table.py [input] [output]\n\n')
    sys.stderr.write('converts a table from one format to another.\n' \
        'The format is specified in the file name.\n\n')
    sys.exit(-1)

r = teca_table_reader.New()
r.set_file_name(sys.argv[1])

w = teca_table_writer.New()
w.set_input_connection(r.get_output_port())
w.set_file_name(sys.argv[2])

w.update()
