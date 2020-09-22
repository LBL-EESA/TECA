from teca import *
import sys, os


if len(sys.argv) != 2:
    sys.stderr.write('Usage: test_table_from_stream [baseline]\n')
    sys.exit(-1)

baseline = sys.argv[1]

#bname = os.basename(baseline)
bname, bext = os.path.splitext(os.path.basename(baseline))

# write intermediate results in the other format
oext = '.csv' if bext == '.bin' else '.bin'
oname = bname + oext;

sys.stderr.write('reading(%s) ---> writing(%s)\n'%(bname+bext, oname))

br = teca_table_reader.New()
br.set_file_name(baseline)

bw = teca_table_writer.New()
bw.set_input_connection(br.get_output_port())
bw.set_file_name(oname)
bw.update()

# read that back in and diff
sys.stderr.write('reading(%s) ---> diff(%s)\n'%(oname, bname+bext))

tr = teca_table_reader.New()
tr.set_file_name(oname)

diff = teca_dataset_diff.New()
diff.set_input_connection(0, br.get_output_port())
diff.set_input_connection(1, tr.get_output_port())
diff.update()

sys.stderr.write('OK\n\n')
