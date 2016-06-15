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

t1[2,1] = 'Sat'
t1[3,1] = 'Sun'

sys.stderr.write('dumping column 1...\n')
r = 0
c = 1
while r < 5:
    sys.stderr.write('t1[%d,%d] = %s\n'%(r,c,str(t1[r,c])))
    r += 1

t2 = teca_table.New()
t2.declare_column('summary', 's')
t2.set_column('summary', ['a','b','c','d','e','f'])

sys.stderr.write('dumping table contents...\n')
sys.stderr.write('%s\n'%(str(t2)))

sys.stderr.write('packaging tables into a database...\n')
db = teca_database.New()
db.append_table('summary', t2)
db.append_table('details', t1)

sys.stderr.write('dumping the database...\n')
sys.stderr.write('%s\n'%(str(db)))


sys.stderr.write('writing table to disk...\n')
def serve_table(port, data, req):
    global t1
    return t1

tab_serv = teca_programmable_algorithm.New()
tab_serv.set_number_of_input_connections(0)
tab_serv.set_execute_callback(serve_table)

tab_wri = teca_table_writer.New()
tab_wri.set_input_connection(tab_serv.get_output_port())
tab_wri.set_file_name('table_%t%.bin')
tab_wri.update()


sys.stderr.write('writing database to disk...\n')
def serve_database(port, data, req):
    global db
    return db

db_serv = teca_programmable_algorithm.New()
db_serv.set_number_of_input_connections(0)
db_serv.set_execute_callback(serve_database)

db_wri = teca_table_writer.New()
db_wri.set_input_connection(db_serv.get_output_port())
db_wri.set_file_name('database_%s%_%t%.bin')
db_wri.update()
