import sys
import numpy as np
from teca import *

set_stack_trace_on_error()

t1 = teca_table.New()
t1.declare_columns(['event','day','strength','magnitude'], ['i','s','f','d'])
t1.declare_column('flag', 'ull')

c1 = np.array([1,2,3,4,5], dtype=np.int)
c2 = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri']
c3 = np.array([1.2,2.3,3.4,4.5,5.6], dtype=np.double)
c4 = np.array([6,7,8,9,10], dtype=np.double)
c6 = np.array([0,1,0,1,0], dtype=np.int)

t1.set_column('event', np.array([1,2,3,4,5], dtype=np.int))
t1.set_column('day', ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri'])
t1.set_column('strength', np.array([1.2,2.3,3.4,4.5,5.6], dtype=np.double))
t1.set_column('magnitude', np.array([6,7,8,9,10], dtype=np.double))
t1.set_column('flag', np.array([0,1,0,1,0], dtype=np.int))

sys.stderr.write('dumping table contents...\n')
sys.stderr.write('%s\n'%(str(t1)))
sys.stderr.write('Ok!\n')

t1[2,1] = 'Sat'
t1[3,1] = 'Sun'

sys.stderr.write('dumping column 1...\n')
r = 0
c = 1
while r < 5:
    sys.stderr.write('t1[%d,%d] = %s\n'%(r,c,str(t1[r,c])))
    r += 1
sys.stderr.write('Ok!\n')

t2 = teca_table.New()
t2.declare_column('summary', 's')
t2.set_column('summary', ['a','b','c','d','e','f'])

sys.stderr.write('dumping table contents...\n')
sys.stderr.write('%s\n'%(str(t2)))
sys.stderr.write('Ok!\n')

sys.stderr.write('packaging tables into a database...')
db = teca_database.New()
db.append_table('summary', t2)
db.append_table('details', t1)
sys.stderr.write('Ok!\n')

sys.stderr.write('dumping the database...\n')
sys.stderr.write('%s\n'%(str(db)))
sys.stderr.write('Ok!\n')


sys.stderr.write('writing 2 tables to disk...')
tab_serv = teca_dataset_source.New()
tab_serv.append_dataset(t1)
tab_serv.append_dataset(t2)

tab_wri = teca_table_writer.New()
tab_wri.set_input_connection(tab_serv.get_output_port())
tab_wri.set_file_name('table_%t%.bin')
tab_wri.set_executive(teca_index_executive.New())
tab_wri.update()
sys.stderr.write('Ok!\n')


sys.stderr.write('writing database containing 2 tables to disk...')
db_serv = teca_dataset_source.New()
db_serv.append_dataset(db)

db_wri = teca_table_writer.New()
db_wri.set_input_connection(db_serv.get_output_port())
db_wri.set_file_name('database_%s%_%t%.bin')
db_wri.set_executive(teca_index_executive.New())
db_wri.update()
sys.stderr.write('Ok!\n')
