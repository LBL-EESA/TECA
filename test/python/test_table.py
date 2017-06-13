from teca import *
import sys,os,numpy as np

set_stack_trace_on_error()

if len(sys.argv) < 2:
    sys.stderr.write('test_teca_table [baseline]\n')
    sys.exit(-1)

baseline = sys.argv[1]

tab = teca_table.New()
tab.declare_columns(['A(f)','B(d)','C(i)','D(l)'],['f','d','i','l'])
tab << np.float32(1.1) << np.float64(2.1) << np.int8(31) << np.int16(41) \
    << np.int32(12) << np.int64(22) << np.uint8(32) << np.uint16(42) \
    << np.uint32(13) << np.uint64(23) << np.long(33) << np.byte(43) \
    << -14 << -24 << -34 << -44 \
    << (-.15, -2.5, -35, -45) \
    << [1.6, 2.6, 36, 46] \
    << np.array([1.7, 2.7, 37, 47], dtype=np.float32) \
    << np.array([1.8, 2.8, 38, 48], dtype=np.float64) \
    << np.array([19, 29, 39, 49], dtype=np.int64) \
    << np.array([110, 210, 310, 410], dtype=np.int32) \
    << np.array([111, 211, 311, 411], dtype=np.int16) \
    << np.array([112, 212, 312, 412], dtype=np.int8) \
    << np.array([113, 213, 313, 413], dtype=np.uint64) \
    << np.array([114, 214, 314, 414], dtype=np.uint32) \
    << np.array([115, 215, 315, 415], dtype=np.uint16) \
    << np.array([116, 216, 316, 416], dtype=np.uint8)

tab[0,0] = np.float32(-tab[0,0])
tab[1,1] = np.float64(-tab[1,1])
tab[2,2] = -tab[2,2]
tab[3,3] = -tab[3,3]

source = teca_dataset_source.New()
source.set_dataset(tab)

if os.path.exists(baseline):
    # run the test
    table_reader = teca_table_reader.New()
    table_reader.set_file_name(baseline)
    diff = teca_dataset_diff.New()
    diff.set_input_connection(0, table_reader.get_output_port())
    diff.set_input_connection(1, source.get_output_port())
    diff.update()
else:
    # write the data
    sys.stderr.write('%s\n'%(str(tab)))
    table_writer = teca_table_writer.New()
    table_writer.set_input_connection(source.get_output_port())
    table_writer.set_file_name(baseline)
    table_writer.update();
