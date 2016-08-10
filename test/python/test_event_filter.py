from teca import *
import sys
import os

argc = len(sys.argv)
if argc < 3:
    sys.stderr.write('test_event_filter.py [input] [output]\n')
    sys.exit(-1)

table_in = sys.argv[1]
table_out = sys.argv[2]

r = teca_table_reader.New()
r.set_file_name(table_in)

f = teca_event_filter.New()
f.set_input_connection(r.get_output_port())
f.set_x_coordinate_column('lon')
f.set_y_coordinate_column('lat')
f.set_region_x_coordinates([180, 180, 270, 270, 180])
f.set_region_y_coordinates([-10, 10, 10, -10, -10])
f.set_region_sizes([5])
f.set_time_column('time')
f.set_start_time(4196.24)
f.set_end_time(4196.38)

if os.path.exists(table_out):
    #run the test
    br = teca_table_reader.New()
    br.set_file_name(table_out)
    d = teca_dataset_diff.New()
    d.set_input_connection(0, br.get_output_port())
    d.set_input_connection(1, f.get_output_port())
    d.update()
else:
    # write the data
    w = teca_table_writer.New()
    w.set_input_connection(f.get_output_port())
    w.set_file_name(table_out)
    w.update();

sys.exit(0)
