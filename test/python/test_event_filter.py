from teca import *
import sys
import os

argc = len(sys.argv)
if argc < 3:
    sys.stderr.write('test_table_region_mask.py [input] [output]\n')
    sys.exit(-1)

table_in = sys.argv[1]
table_out = sys.argv[2]

tr = teca_table_reader.New()
tr.set_file_name(table_in)

rm = teca_table_region_mask.New()
rm.set_input_connection(tr.get_output_port())
rm.set_x_coordinate_column('lon')
rm.set_y_coordinate_column('lat')
rm.set_region_x_coordinates([180, 180, 270, 270, 180])
rm.set_region_y_coordinates([-10, 10, 10, -10, -10])
rm.set_region_sizes([5])
rm.set_result_column('in_spatial')

ee = teca_evaluate_expression.New()
ee.set_input_connection(rm.get_output_port())
ee.set_expression('((time > 4196.23) && (time < 4196.39))')
ee.set_result_variable('in_temporal')

rr = teca_table_remove_rows.New()
rr.set_input_connection(ee.get_output_port())
rr.set_mask_expression('!(in_temporal && in_spatial)')
rr.set_remove_dependent_variables(1)

if os.path.exists(table_out):
    #run the test
    br = teca_table_reader.New()
    br.set_file_name(table_out)
    d = teca_dataset_diff.New()
    d.set_input_connection(0, br.get_output_port())
    d.set_input_connection(1, rr.get_output_port())
    d.update()
else:
    # write the data
    w = teca_table_writer.New()
    w.set_input_connection(rr.get_output_port())
    w.set_file_name(table_out)
    w.update();

sys.exit(0)
