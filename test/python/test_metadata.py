from teca import *
import numpy as np
import sys

set_stack_trace_on_error()

def start_sec(s):
    sys.stderr.write('checking '+s+'...\n')

def end_sec(s):
    sys.stderr.write('checking '+s+'...ok\n\n')

start_sec('construction')
md = teca_metadata()
end_sec('construction')

start_sec('empty')
sys.stderr.write('%s\n'%('yes' if md.empty() else 'no'))
end_sec('empty')

start_sec('set from object')
md['int'] = 1
md['float'] = 1.0
md['string'] = 'other string'
md['bool'] = True
end_sec('set from object')

start_sec('set from list')
md['int list'] = [0, 1, 2, 3, 4, 5]
md['float list'] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
md['string list'] = ['apple', 'boat', 'cat', 'diamond']
md['bool list'] = [True, False]
end_sec('set from list')

start_sec('set from numpy')
md['int32 array'] = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
md['int64 array'] = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
md['uint32 array'] = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32)
md['uint64 array'] = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint64)
md['float32 array'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
md['float64 array'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
end_sec('set from numpy')

start_sec('get')
md_keys = ['int', 'float', 'string', 'bool', 'int list', \
    'float list', 'string list', 'bool list', \
    'int32 array', 'int64 array', 'uint32 array', \
     'uint64 array', 'float32 array', 'float64 array']
for md_key in md_keys:
    sys.stderr.write('%s = %s\n'%(md_key, str(md[md_key])))
end_sec('get')

start_sec('empty')
sys.stderr.write('%s\n'%('yes' if md.empty() else 'no'))
end_sec('empty')

start_sec('str')
sys.stderr.write('md = {%s}\n'%(str(md)))
end_sec('str')

start_sec('clear')
md.clear()
end_sec('clear')

start_sec('destruct')
del md
end_sec('destruct')
