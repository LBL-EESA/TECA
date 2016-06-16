from teca import *
import numpy as np
import sys

set_stack_trace_on_error()

def start_sec(s):
    sys.stderr.write('checking '+s+'...\n')

def end_sec(s):
    sys.stderr.write('checking '+s+'...ok\n\n')

def out(s):
    sys.stderr.write(s)

start_sec('constructor')
col = teca_array_collection.New()
end_sec('constructor')

start_sec('set')
col['a'] = np.array([0,1,2,3,4,5])
col.set('b', teca_variant_array.New([5,6,7,8,9]))
col.set('b', teca_variant_array.New([105,106,107,108,109]))
end_sec('set')

start_sec('append')
col.append('c', teca_variant_array.New([5,6,7,8,9]))
col.append('c', teca_variant_array.New([105,106,107,108,109]))
end_sec('append')

start_sec('get')
out('a = %s\n'%str(col['a']))
out('b = %s\n'%str(col['b']))
out('c = %s\n'%str(col.get('c')))
end_sec('get')

start_sec('str')
out('%s\n'%str(col))
end_sec('str')
